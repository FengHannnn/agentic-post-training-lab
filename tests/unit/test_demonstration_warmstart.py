import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from demonstration_warmstart_common import choose_batches,encode_demonstration,compare_metrics
from train_demonstration_warmstart import sft_step

def groups(values):
    return [[SimpleNamespace(group_id=str(g),sample_index=i,reward=v) for i,v in enumerate(row)]
            for g,row in enumerate(values)]

def test_mixed_only_frozen_order_one_pass_and_no_signal():
    g=groups([[0]*32,[0,1]*16,[1]*32]+[[0,1]*16 for _ in range(5)])
    batches,audit=choose_batches(g,[str(i) for i in range(8)])
    assert batches==[['1','3','4','5'],['6','7']]
    assert len([x for b in batches for x in b])==6
    assert [r['mixed'] for r in audit]==[False,True,False,True,True,True,True,True]
    assert choose_batches(groups([[0]*32]),['0'])[0]==[]

@pytest.mark.parametrize('mutation',['fraction','nan','duplicate','short','order'])
def test_invalid_fresh_binary_groups_rejected(mutation):
    g=groups([[0,1]*16,[0,1]*16])
    if mutation=='fraction':g[0][0].reward=.3
    elif mutation=='nan':g[0][0].reward=float('nan')
    elif mutation=='duplicate':g[0][1].sample_index=0
    elif mutation=='short':g[0].pop()
    else:g.reverse()
    with pytest.raises(ValueError):choose_batches(g,['0','1'])

def test_demonstration_masks_prompt_and_includes_real_eos():
    from agentic_post_training.evaluation.tool_calls import EvalCase
    text='{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}'
    case=EvalCase('case','math','add','tool_call','1+2',3)
    row=SimpleNamespace(chosen=text,target=lambda:case)
    class Tokenizer:
        eos_token_id=0
        def apply_chat_template(self,*a,**k):return [4,5]
        def encode(self,text,**k):return [6,7]
        def decode(self,*a,**k):return text
    result=encode_demonstration(row,Tokenizer())
    assert result['labels']==[-100,-100,6,7,0]
    assert result['loss_mask']==[0,0,1,1,1]
    row.chosen=text.replace('1+2','1-2')
    with pytest.raises(ValueError,match='correct'):encode_demonstration(row,Tokenizer())

def test_supervised_gain_is_not_reported_as_grpo_gain():
    a={'development':dict(strict=1,verified_correct=2,completed_semantic=2,length_stops=0)}
    b={'development':dict(strict=4,verified_correct=5,completed_semantic=5,length_stops=0)}
    assert compare_metrics(a,b)['development']['verified_correct']==3
    assert compare_metrics(b,b)['development']['verified_correct']==0

def test_real_lora_sft_then_fresh_sampling_grpo_carry31_and_resume(tmp_path):
    from transformers import LlamaConfig,LlamaForCausalLM
    from peft import LoraConfig,get_peft_model
    from agentic_post_training.trainers.dpo import parameter_digest
    from agentic_post_training.trainers.grpo import (lora_parameters,guarded_step,save_optimizer_checkpoint,
        restore_optimizer_checkpoint,adapter_snapshot,optimizer_snapshot,compare_adapter_snapshot,compare_optimizer_snapshot)
    from agentic_post_training.rollouts.records import RolloutSample
    from agentic_post_training.rollouts.trainer_scores import score_response
    from agentic_post_training.algorithms.losses import group_relative_advantages
    from agentic_post_training.trainers.policy_iteration import file_hash
    from reward_comparison_common import import_comparison_adam
    torch.set_num_threads(1);torch.manual_seed(91)
    policy=get_peft_model(LlamaForCausalLM(LlamaConfig(vocab_size=16,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.,
        pad_token_id=0,eos_token_id=0,_attn_implementation='eager')),
        LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    row={'input_ids':[1,2,5,0],'attention_mask':[1]*4,'labels':[-100,-100,5,0],'loss_mask':[0,0,1,1]}
    frozen=parameter_digest(policy,frozen_only=True)
    parent_opt=torch.optim.AdamW(lora_parameters(policy),lr=1e-5,weight_decay=0.)
    for _ in range(31):sft_step(policy,parent_opt,[row],0)
    save_optimizer_checkpoint(policy,parent_opt,tmp_path/'parent',15,'old')
    parent_digest=parameter_digest(policy)
    parent_moments=optimizer_snapshot(parent_opt)
    reference=copy.deepcopy(policy).requires_grad_(False).eval()
    sft_opt=torch.optim.AdamW(lora_parameters(policy),lr=5e-5,weight_decay=0.)
    sft_step(policy,sft_opt,[row],0);sft_step(policy,sft_opt,[row],0)
    warm_digest=parameter_digest(policy)
    assert warm_digest!=parent_digest and parameter_digest(reference)==parent_digest
    # Draw real new tokens from B. Synthetic parity rewards test the training plumbing only.
    with torch.no_grad():
        logits=policy(input_ids=torch.tensor([[1,2]])).logits[0,-1]
        logps=logits.log_softmax(-1)
        tokens=torch.multinomial(logps.exp(),32,replacement=True).tolist()
    samples=[RolloutSample('g','g',i,91,[1,2],[t],[logps[t].item()],[logps[t].item()],
        0,'eos' if t==0 else 'length',str(t),str(t)) for i,t in enumerate(tokens)]
    rewards=[float(t%2) for t in tokens]
    assert len(set(rewards))==2
    advantages=group_relative_advantages(torch.tensor([rewards]))[0].tolist()
    targets={};refs={}
    for s,reward,adv in zip(samples,rewards,advantages):
        key=s.group_id,s.sample_index
        targets[key]={'reward':reward,'advantage':adv,
            'trainer_old_token_log_probs':score_response(policy,[1,2],s.response_token_ids,grad_enabled=False).tolist()}
        refs[key]=score_response(reference,[1,2],s.response_token_ids,grad_enabled=False)
    optimizer=torch.optim.AdamW(lora_parameters(policy),lr=1e-5,weight_decay=0.)
    import_comparison_adam(policy,optimizer,tmp_path/'parent/optimizer.pt',
        expected_hash=file_hash(tmp_path/'parent/optimizer.pt'),old_binding='old',new_binding='new')
    assert compare_optimizer_snapshot(optimizer,parent_moments,0.)['optimizer_state_passed']
    save_optimizer_checkpoint(policy,optimizer,tmp_path/'initial',0,'new')
    initial_adapter=adapter_snapshot(policy)
    config=dict(beta=.02,clip_epsilon=.2,tis_cap=2.,max_grad_norm=1.,stop_sampled_k3=.05,stop_ratio_clip_fraction=.5)
    result=guarded_step(policy,optimizer,samples,targets,refs,config)
    expected_a=adapter_snapshot(policy);expected_o=optimizer_snapshot(optimizer)
    assert result['updated'] and all(v['step'].item()==32 for v in optimizer.state.values())
    with torch.no_grad():
        for name,p in policy.named_parameters():
            if name in initial_adapter:p.copy_(initial_adapter[name])
    assert restore_optimizer_checkpoint(policy,optimizer,tmp_path/'initial','new')==0
    replay=guarded_step(policy,optimizer,samples,targets,refs,config)
    assert replay['updated'] and replay['loss']==pytest.approx(result['loss'],abs=1e-7)
    assert compare_adapter_snapshot(policy,expected_a,1e-7)['passed']
    assert compare_optimizer_snapshot(optimizer,expected_o,1e-7)['optimizer_state_passed']
    assert parameter_digest(policy,frozen_only=True)==frozen
    assert parameter_digest(policy)!=warm_digest

@pytest.mark.parametrize('failed_stage',[0,1,2])
def test_driver_stops_and_packages_failure_evidence(tmp_path,monkeypatch,failed_stage):
    import zipfile
    import run_demonstration_warmstart as driver
    stub=tmp_path/'stub';stub.write_text('data')
    manifest=tmp_path/'bundle.json'
    from agentic_post_training.trainers.policy_iteration import file_hash
    manifest.write_text(json.dumps({'files':{'stub':file_hash(stub)},'required_existing_files':{'stub':file_hash(stub)}}))
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    monkeypatch.setattr(driver,'OUTPUT',tmp_path/'run')
    monkeypatch.setattr(driver,'ARCHIVE',tmp_path/'review.zip')
    monkeypatch.setattr(driver,'BUNDLE',manifest)
    monkeypatch.setattr(driver,'preflight',lambda:{'passed':True})
    calls=[]
    def spawn(*args,**kwargs):
        index=len(calls);calls.append(args[0])
        return SimpleNamespace(wait=lambda **kw:2 if index==failed_stage else 0)
    monkeypatch.setattr(driver.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError):driver.dispatch(2.18)
    assert len(calls)==failed_stage+1
    with zipfile.ZipFile(tmp_path/'review.zip') as z:
        run=json.loads(z.read('run/cost_and_run.json'))
        assert run['error'] and not run['automatic_retry']
    with pytest.raises(FileExistsError):driver.dispatch(2.18)
