import copy

import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig,PeftModel,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM

from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import score_response
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (guarded_step,accumulate_minibatch,lora_parameters,
    save_optimizer_checkpoint,restore_optimizer_checkpoint,adapter_snapshot,compare_adapter_snapshot,
    optimizer_snapshot,compare_optimizer_snapshot)


@pytest.fixture
def tiny():
    threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(42)
    config=LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=2,num_key_value_heads=2,attention_dropout=0.)
    config._attn_implementation='eager'
    base=LlamaForCausalLM(config);base_state=copy.deepcopy(base.state_dict())
    model=get_peft_model(base,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],
        init_lora_weights=False,task_type='CAUSAL_LM')).eval()
    reference=copy.deepcopy(model).requires_grad_(False).eval()
    samples=[];targets={};cache={}
    for i in range(8):
        tokens=[4+i,2]
        old=score_response(model,[1,3],tokens,grad_enabled=False)
        s=RolloutSample('group','group',i,42+i,[1,3],tokens,old.tolist(),old.tolist(),2,'eos','x','x<EOS>')
        samples.append(s);targets['group',i]={'trainer_old_token_log_probs':old.tolist(),'advantage':1. if i<4 else -1.,'reward':float(i<4)}
        cache['group',i]=score_response(reference,[1,3],tokens,grad_enabled=False)
    cfg={'beta':.02,'clip_epsilon':.2,'tis_cap':2.,'max_grad_norm':1.,'stop_sampled_k3':.05,'stop_ratio_clip_fraction':.5}
    yield model,reference,samples,targets,cache,cfg,config,base_state
    torch.set_num_threads(threads)


def test_real_lora_update_only_changes_adapter_and_resume_reproduces_next_step(tiny,tmp_path):
    model,ref,samples,targets,cache,cfg,model_config,base_state=tiny
    before=parameter_digest(model);frozen=parameter_digest(model,frozen_only=True);ref_before=parameter_digest(ref)
    opt=torch.optim.AdamW(lora_parameters(model),lr=1e-3,weight_decay=0.)
    first=guarded_step(model,opt,samples,targets,cache,cfg)
    assert first['updated'] and first['grad_norm_before_clip']>0 and parameter_digest(model)!=before
    assert parameter_digest(model,frozen_only=True)==frozen and parameter_digest(ref)==ref_before
    assert all(p.grad is None for p in ref.parameters())
    save_optimizer_checkpoint(model,opt,tmp_path/'checkpoint',1,'bound-input')
    second=guarded_step(model,opt,samples,targets,cache,cfg)
    assert second['updated']
    expected=adapter_snapshot(model);expected_optimizer=optimizer_snapshot(opt)
    base=LlamaForCausalLM(model_config);base.load_state_dict(base_state)
    restored=PeftModel.from_pretrained(base,str(tmp_path/'checkpoint/adapter'),is_trainable=True,autocast_adapter_dtype=False).eval()
    restored_opt=torch.optim.AdamW(lora_parameters(restored),lr=1e-3,weight_decay=0.)
    assert restore_optimizer_checkpoint(restored,restored_opt,tmp_path/'checkpoint','bound-input')==1
    resumed=guarded_step(restored,restored_opt,samples,targets,cache,cfg)
    assert resumed['updated'] and resumed['loss']==pytest.approx(second['loss'],abs=1e-7)
    assert compare_adapter_snapshot(restored,expected,1e-7)['passed']
    assert compare_optimizer_snapshot(restored_opt,expected_optimizer,1e-8)['optimizer_state_passed']
    assert parameter_digest(restored,frozen_only=True)==frozen
    with pytest.raises(ValueError,match='checkpoint input/parameter order differs'):
        restore_optimizer_checkpoint(restored,restored_opt,tmp_path/'checkpoint','wrong-input')


def test_token_weighted_microbatch_gradient_matches_global_token_mean(tiny):
    model,ref,samples,targets,cache,cfg,_,_=tiny
    samples=samples[:2];other=copy.deepcopy(model)
    # Unequal response lengths exercise the token-count denominator.
    s=samples[1];tokens=[7,8,2];old=score_response(model,s.prompt_token_ids,tokens,grad_enabled=False)
    samples[1]=RolloutSample(s.group_id,s.case_id,s.sample_index,s.seed,s.prompt_token_ids,tokens,
        old.tolist(),old.tolist(),2,'eos','x','x<EOS>')
    targets['group',1]={'trainer_old_token_log_probs':old.tolist(),'advantage':-1.,'reward':0.}
    cache['group',1]=old.clone()
    model.zero_grad(set_to_none=True)
    metrics=accumulate_minibatch(model,samples,targets,cache,beta=0.,clip_epsilon=.2,tis_cap=2.)
    new_values=[];old_values=[];mask=[]
    for s in samples:
        new=score_response(other,s.prompt_token_ids,s.response_token_ids,grad_enabled=True)
        old=torch.tensor(targets[s.group_id,s.sample_index]['trainer_old_token_log_probs'])
        new_values.append(F.pad(new,(0,3-len(new))));old_values.append(F.pad(old,(0,3-len(old))))
        mask.append([True]*len(new)+[False]*(3-len(new)))
    new=torch.stack(new_values).unsqueeze(0);old=torch.stack(old_values).unsqueeze(0)
    loss,_=corrected_grpo_policy_loss(new,old,old,torch.tensor([[1.,-1.]]),torch.tensor([mask]))
    loss.backward()
    assert metrics['loss']==pytest.approx(loss.item(),abs=1e-6)
    for (n,p),(n2,q) in zip(model.named_parameters(),other.named_parameters()):
        if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_clip_stop_discards_gradients_without_optimizer_update(tiny):
    model,ref,samples,targets,cache,cfg,_,_=tiny
    for row in targets.values():row['trainer_old_token_log_probs']=[v-.5 for v in row['trainer_old_token_log_probs']]
    before=parameter_digest(model);opt=torch.optim.AdamW(lora_parameters(model),lr=1e-3)
    result=guarded_step(model,opt,samples,targets,cache,cfg)
    assert not result['updated'] and result['ratio_clip_fraction']==1
    assert parameter_digest(model)==before and all(p.grad is None for p in model.parameters()) and not opt.state
