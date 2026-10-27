import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.integration.test_grpo_first_update import tiny
from agentic_post_training.rollouts.batched_scores import score_responses
from agentic_post_training.rollouts.trainer_scores import score_response
from agentic_post_training.trainers.grpo import accumulate_minibatch,guarded_step,lora_parameters,adapter_snapshot,compare_adapter_snapshot
from agentic_post_training.trainers.grpo_microbatch import accumulate_microbatches,guarded_microbatch_step
from agentic_post_training.trainers.dpo import parameter_digest


def variable_samples(tiny):
    model,ref,samples,targets,cache,cfg,_,_=tiny
    varied=[]
    for i,s in enumerate(samples):
        prompt=[1]+[3]*(1+i%3)
        response=[4+i]*(1+i%3)+[2]
        old=score_response(model,prompt,response,grad_enabled=False)
        q=old-.01
        varied.append(replace(s,prompt_token_ids=prompt,response_token_ids=response,
            policy_token_log_probs=q.tolist(),behavior_token_log_probs=q.tolist()))
        targets[s.group_id,i]['trainer_old_token_log_probs']=old.tolist()
        cache[s.group_id,i]=old.clone()
    with torch.no_grad():
        for n,p in model.named_parameters():
            if p.requires_grad:p.add_(.003)
    return model,varied,targets,cache,cfg


def test_complete_benchmark_pipeline_restores_candidate_and_writes_all_probes(tiny,tmp_path,monkeypatch):
    from tests.unit.test_grpo_microbatch_benchmark import runner
    from agentic_post_training.trainers.grpo import save_optimizer_checkpoint
    module=runner();model,reference,samples,targets,_,cfg,_,_=tiny
    expanded=[];expanded_targets={}
    for group in range(4):
        for s in samples:
            row=replace(s,group_id=f'group{group}',case_id=f'group{group}')
            expanded.append(row);expanded_targets[row.group_id,row.sample_index]=targets[s.group_id,s.sample_index]
    source=tmp_path/'input';source.mkdir()
    opt=torch.optim.AdamW(lora_parameters(model),lr=1e-5,weight_decay=0.)
    for p in lora_parameters(model):
        opt.state[p]={'step':torch.tensor(16.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    save_optimizer_checkpoint(model,opt,source/'final-checkpoint',16,'bound')
    module.write(source/'summary.json',{'fixture':True})
    summary={'engineering_checks_passed':True,'logical_optimizer_steps':16,'files_sha256':{},
        'final_parameter_sha256':parameter_digest(model),'initial_parameter_sha256':parameter_digest(reference)}
    manifest={'source_sha256':{},'checkpoint_input_digest':'bound'}
    training={**cfg,'learning_rate':1e-5}
    monkeypatch.setattr(module,'load_inputs',lambda:(module.read(module.PLAN),source,summary,manifest,
        training,{},tmp_path/'baseline',Path('unused'),SimpleNamespace(pad_token_id=2),expanded,expanded_targets))
    monkeypatch.setattr(module.first,'load_model',lambda base,adapter,trainable:copy.deepcopy(model if trainable else reference))
    monkeypatch.setattr(module.first,'check_gpu',lambda cfg:'CPU test double')
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    output=tmp_path/'benchmark'
    module.benchmark(output)
    result=module.read(output/'summary.json')
    assert result['benchmark_completed'] and result['candidate_restored_to_input_checkpoint']
    assert result['temporary_optimizer_steps_discarded']==12 and result['published_candidate_updates']==0
    assert all(row['passed'] for row in result['correctness'].values())
    assert len(module.read(output/'timing.json'))==9
    assert len(list(output.glob('correctness-size*.json')))==3
    assert len(list(output.glob('timing-repeat*.json')))==9


def test_batched_original_token_scores_equal_serial_with_unequal_lengths_and_eos_as_pad(tiny):
    model,samples,_,_,_=variable_samples(tiny)
    expected=[score_response(model,s.prompt_token_ids,s.response_token_ids,grad_enabled=False) for s in samples]
    observed={}
    def inspect(module,args,kwargs):observed.update(ids=kwargs['input_ids'].detach().clone(),mask=kwargs['attention_mask'].detach().clone())
    hook=model.register_forward_pre_hook(inspect,with_kwargs=True)
    actual=score_responses(model,[s.prompt_token_ids for s in samples],[s.response_token_ids for s in samples],2,grad_enabled=True)
    hook.remove()
    for i,(s,left,right) in enumerate(zip(samples,expected,actual)):
        torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
        assert right.requires_grad and len(right)==len(s.response_token_ids)
        length=len(s.prompt_token_ids)+len(s.response_token_ids)
        assert observed['mask'][i,length-1]==1 and observed['ids'][i,length-1]==2
        assert (observed['mask'][i,length:]==0).all()


@pytest.mark.parametrize('size',[1,2,4,8])
def test_microbatch_gradient_matches_serial_with_token_denominator_and_kl(tiny,size):
    model,samples,targets,cache,cfg=variable_samples(tiny);other=copy.deepcopy(model)
    original=accumulate_minibatch(model,samples,targets,cache,beta=.02,clip_epsilon=.2,tis_cap=2.)
    forwards=[];hook=other.register_forward_hook(lambda *args:forwards.append(1))
    actual=accumulate_microbatches(other,samples,targets,cache,microbatch_size=size,pad_token_id=2,
        beta=.02,clip_epsilon=.2,tis_cap=2.)
    hook.remove()
    assert actual['forward_backward_calls']==len(forwards)==(8+size-1)//size
    assert actual['loss']==pytest.approx(original['loss'],abs=1e-6)
    assert actual['response_tokens']==original['response_tokens']
    for (n,p),(m,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
        else:assert p.grad is None and q.grad is None


def test_one_optimizer_step_matches_and_last_partial_chunk_is_retained(tiny):
    model,samples,targets,cache,cfg=variable_samples(tiny);samples=samples[:7];other=copy.deepcopy(model)
    frozen=parameter_digest(model,frozen_only=True)
    opt=torch.optim.AdamW(lora_parameters(model),lr=1e-5,weight_decay=0.)
    opt2=torch.optim.AdamW(lora_parameters(other),lr=1e-5,weight_decay=0.)
    left=guarded_step(model,opt,samples,targets,cache,cfg)
    right=guarded_microbatch_step(other,opt2,samples,targets,cache,cfg,microbatch_size=4,pad_token_id=2)
    assert left['updated'] and right['updated'] and right['forward_backward_calls']==2 and right['completions']==7
    assert compare_adapter_snapshot(other,adapter_snapshot(model),1e-6)['passed']
    assert parameter_digest(other,frozen_only=True)==frozen


def test_microbatch_stop_discards_all_gradients(tiny):
    model,samples,targets,cache,cfg=variable_samples(tiny)
    for r in targets.values():r['trainer_old_token_log_probs']=[x-.5 for x in r['trainer_old_token_log_probs']]
    before=parameter_digest(model);opt=torch.optim.AdamW(lora_parameters(model),lr=1e-5)
    result=guarded_microbatch_step(model,opt,samples,targets,cache,cfg,microbatch_size=8,pad_token_id=2)
    assert not result['updated'] and parameter_digest(model)==before
    assert all(p.grad is None for p in model.parameters()) and not opt.state


@pytest.mark.parametrize('kind',['empty','mismatched','empty_response','training','bad_pad'])
def test_invalid_batch_scoring_inputs_rejected(tiny,kind):
    model,*_=tiny;prompts=[[1]];responses=[[4,2]];pad=2
    if kind=='empty':prompts=[];responses=[]
    elif kind=='mismatched':responses=[]
    elif kind=='empty_response':responses=[[]]
    elif kind=='training':model.train()
    else:pad=-1
    with pytest.raises(ValueError):score_responses(model,prompts,responses,pad,grad_enabled=True)
