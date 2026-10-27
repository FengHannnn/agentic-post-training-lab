"""A bounded single-rollout PPO/GRPO update with detached old/q/reference scores.

Token TIS and sampled k3 are approximate surrogates on stored rollout histories;
neither is claimed to correct the whole trajectory/group distribution.
"""

import math
from pathlib import Path

import torch

from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss, token_tis_weights
from agentic_post_training.rollouts.trainer_scores import score_response


def completion_objective(new, old, behavior, reference, advantage, *, beta=.02, clip_epsilon=.2, tis_cap=2.):
    if (new.ndim!=1 or not new.numel() or reference.shape!=new.shape or reference.requires_grad
        or not math.isfinite(beta) or beta<0 or not math.isfinite(advantage)):
        raise ValueError("aligned fixed reference, finite advantage and nonnegative beta required")
    if not torch.isfinite(reference).all() or (reference>1e-6).any():
        raise ValueError("invalid reference log probabilities")
    mask=torch.ones((1,1,new.numel()),dtype=torch.bool,device=new.device)
    shaped=lambda x:x.reshape(1,1,-1)
    adv=torch.tensor([[advantage]],dtype=new.dtype,device=new.device)
    policy_loss,stats=corrected_grpo_policy_loss(shaped(new),shaped(old),shaped(behavior),adv,mask,
        upper_cap=tis_cap,clip_epsilon=clip_epsilon)
    weights,_=token_tis_weights(shaped(old),shaped(behavior),mask,upper_cap=tis_cap)
    delta=reference.double()-new.double()
    k3=torch.expm1(delta)-delta
    if not torch.isfinite(k3).all() or (k3<0).any():
        raise ValueError("sampled k3 penalty overflow or invalid value")
    weighted_k3=(weights.flatten()*k3).mean().to(new.dtype)
    loss=policy_loss+beta*weighted_k3
    if not torch.isfinite(loss):raise ValueError("non-finite combined policy/KL loss")
    ratios=(new.detach()-old).exp()
    return loss,{"loss":loss.detach().item(),"policy_loss":policy_loss.detach().item(),
        "sampled_k3":k3.detach().mean().item(),"weighted_sampled_k3":weighted_k3.detach().item(),
        "ratio_clip_fraction":((ratios<1-clip_epsilon)|(ratios>1+clip_epsilon)).float().mean().item(),
        "ratio_min":ratios.min().item(),"ratio_max":ratios.max().item(),**stats}


def accumulate_minibatch(policy, samples, targets, reference_cache, *, beta, clip_epsilon, tis_cap):
    """Accumulate one minibatch token mean using one completion graph at a time."""
    if not samples:raise ValueError("nonempty minibatch required")
    policy.eval()
    device=next(policy.parameters()).device
    total=sum(len(s.response_token_ids) for s in samples)
    totals={};ratio_min=float('inf');ratio_max=0.
    for s in samples:
        key=s.group_id,s.sample_index
        row=targets[key]
        new=score_response(policy,s.prompt_token_ids,s.response_token_ids,grad_enabled=True)
        old=torch.tensor(row['trainer_old_token_log_probs'],dtype=torch.float32,device=device)
        behavior=torch.tensor(s.behavior_token_log_probs,dtype=torch.float32,device=device)
        reference=reference_cache[key].to(device=device,dtype=torch.float32)
        loss,metrics=completion_objective(new,old,behavior,reference,row['advantage'],
            beta=beta,clip_epsilon=clip_epsilon,tis_cap=tis_cap)
        fraction=len(s.response_token_ids)/total
        (loss*fraction).backward()
        for name in ('loss','policy_loss','sampled_k3','weighted_sampled_k3','ratio_clip_fraction','mean_weight','cap_fraction'):
            totals[name]=totals.get(name,0.)+metrics[name]*fraction
        ratio_min=min(ratio_min,metrics['ratio_min']);ratio_max=max(ratio_max,metrics['ratio_max'])
    return {**totals,'ratio_min':ratio_min,'ratio_max':ratio_max,'response_tokens':total,
        'completions':len(samples),'static_rollout_reward_mean':sum(targets[s.group_id,s.sample_index]['reward'] for s in samples)/len(samples)}


def lora_parameters(policy):
    parameters=[p for n,p in policy.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in n for n,p in policy.named_parameters() if p.requires_grad):
        raise ValueError("only LoRA parameters may be trainable")
    if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
        raise ValueError("frozen parameter received gradient")
    return parameters


def guarded_step(policy,optimizer,samples,targets,reference_cache,config):
    optimizer.zero_grad(set_to_none=True)
    metrics=accumulate_minibatch(policy,samples,targets,reference_cache,beta=config['beta'],
        clip_epsilon=config['clip_epsilon'],tis_cap=config['tis_cap'])
    parameters=lora_parameters(policy)
    if metrics['sampled_k3']>config['stop_sampled_k3'] or metrics['ratio_clip_fraction']>config['stop_ratio_clip_fraction']:
        optimizer.zero_grad(set_to_none=True)
        return {**metrics,'updated':False,'stop_reason':'sampled_k3_or_ratio_clip_limit'}
    norm=torch.nn.utils.clip_grad_norm_(parameters,config['max_grad_norm'],error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item()<=0:
        raise ValueError("finite nonzero LoRA gradient required before update")
    optimizer.step()
    return {**metrics,'updated':True,'grad_norm_before_clip':norm.item(),'stop_reason':None}


def save_optimizer_checkpoint(policy,optimizer,path,step,input_digest):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    policy.save_pretrained(path/'adapter',save_embedding_layers=False)
    torch.save({'optimizer':optimizer.state_dict(),'step':step,'input_digest':input_digest,
        'trainable_names':[n for n,p in policy.named_parameters() if p.requires_grad],
        'cpu_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path/'optimizer.pt')


def restore_optimizer_checkpoint(policy,optimizer,path,input_digest):
    # The caller loads policy from this checkpoint's adapter first.
    device=next(policy.parameters()).device
    state=torch.load(Path(path)/'optimizer.pt',map_location=device,weights_only=True)
    if (state['input_digest']!=input_digest or state['trainable_names']!=[n for n,p in policy.named_parameters() if p.requires_grad]):
        raise ValueError("checkpoint input/parameter order differs")
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'].cpu())
    if state['cuda_rng']:
        if not torch.cuda.is_available():raise ValueError("CUDA checkpoint requires CUDA restore")
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    return state['step']


def adapter_snapshot(policy):
    return {n:p.detach().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad}


def compare_adapter_snapshot(policy,snapshot,tolerance=1e-6):
    current=adapter_snapshot(policy)
    if set(current)!=set(snapshot):raise ValueError("adapter parameter names differ")
    error=max((current[k]-snapshot[k]).abs().max().item() for k in current)
    return {'max_adapter_abs_error':error,'passed':error<=tolerance}


def optimizer_snapshot(optimizer):
    state=optimizer.state_dict()
    return {key:{name:value.detach().cpu().clone() if isinstance(value,torch.Tensor) else value
        for name,value in values.items()} for key,values in state['state'].items()}


def compare_optimizer_snapshot(optimizer,snapshot,tolerance=1e-7):
    current=optimizer_snapshot(optimizer)
    if set(current)!=set(snapshot):raise ValueError('optimizer state keys differ')
    error=0.
    for key,values in current.items():
        if set(values)!=set(snapshot[key]):raise ValueError('optimizer buffers differ')
        for name,value in values.items():
            expected=snapshot[key][name]
            if isinstance(value,torch.Tensor):
                if value.shape!=expected.shape:raise ValueError('optimizer tensor shapes differ')
                difference=(value-expected).abs().max().item()
                if name=='step' and difference!=0:raise ValueError('optimizer step counter differs')
                error=max(error,difference)
            elif value!=expected:raise ValueError('optimizer scalar differs')
    return {'max_optimizer_state_abs_error':error,'optimizer_state_passed':error<=tolerance}
