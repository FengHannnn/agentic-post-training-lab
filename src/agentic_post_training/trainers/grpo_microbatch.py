"""Parallel response microbatches with the original effective token denominator."""

import torch

from agentic_post_training.rollouts.batched_scores import score_responses
from agentic_post_training.trainers.grpo import completion_objective,lora_parameters


def accumulate_microbatches(policy,samples,targets,reference_cache,*,microbatch_size,pad_token_id,
                           beta,clip_epsilon,tis_cap,capture_scores=False):
    if not samples or type(microbatch_size) is not int or microbatch_size<1:
        raise ValueError('nonempty samples and positive integer microbatch size required')
    policy.eval();device=next(policy.parameters()).device
    total=sum(len(s.response_token_ids) for s in samples)
    totals={};ratio_min=float('inf');ratio_max=0.;captured=[];calls=0
    for start in range(0,len(samples),microbatch_size):
        chunk=samples[start:start+microbatch_size]
        scores=score_responses(policy,[s.prompt_token_ids for s in chunk],[s.response_token_ids for s in chunk],
            pad_token_id,grad_enabled=True)
        chunk_loss=None
        for s,new in zip(chunk,scores):
            key=s.group_id,s.sample_index;row=targets[key]
            old=torch.tensor(row['trainer_old_token_log_probs'],dtype=torch.float32,device=device)
            q=torch.tensor(s.behavior_token_log_probs,dtype=torch.float32,device=device)
            ref=reference_cache[key].to(device=device,dtype=torch.float32)
            loss,metrics=completion_objective(new,old,q,ref,row['advantage'],
                beta=beta,clip_epsilon=clip_epsilon,tis_cap=tis_cap)
            fraction=len(s.response_token_ids)/total
            term=loss*fraction
            chunk_loss=term if chunk_loss is None else chunk_loss+term
            for name in ('loss','policy_loss','sampled_k3','weighted_sampled_k3','ratio_clip_fraction','mean_weight','cap_fraction'):
                totals[name]=totals.get(name,0.)+metrics[name]*fraction
            ratio_min=min(ratio_min,metrics['ratio_min']);ratio_max=max(ratio_max,metrics['ratio_max'])
            if capture_scores:captured.append(new.detach().cpu().tolist())
        # All response losses in a chunk share one forward graph: backward ONCE.
        chunk_loss.backward();calls+=1
    result={**totals,'ratio_min':ratio_min,'ratio_max':ratio_max,'response_tokens':total,
        'completions':len(samples),'microbatch_size':microbatch_size,'forward_backward_calls':calls,
        'static_rollout_reward_mean':sum(targets[s.group_id,s.sample_index]['reward'] for s in samples)/len(samples)}
    if capture_scores:result['scored_new_log_probs']=captured
    return result


def guarded_microbatch_step(policy,optimizer,samples,targets,reference_cache,config,*,microbatch_size,pad_token_id):
    optimizer.zero_grad(set_to_none=True)
    metrics=accumulate_microbatches(policy,samples,targets,reference_cache,microbatch_size=microbatch_size,
        pad_token_id=pad_token_id,beta=config['beta'],clip_epsilon=config['clip_epsilon'],tis_cap=config['tis_cap'])
    params=lora_parameters(policy)
    if metrics['sampled_k3']>config['stop_sampled_k3'] or metrics['ratio_clip_fraction']>config['stop_ratio_clip_fraction']:
        optimizer.zero_grad(set_to_none=True)
        return {**metrics,'updated':False,'stop_reason':'sampled_k3_or_ratio_clip_limit'}
    norm=torch.nn.utils.clip_grad_norm_(params,config['max_grad_norm'],error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('finite nonzero LoRA gradient required')
    optimizer.step()
    return {**metrics,'updated':True,'grad_norm_before_clip':norm.item(),'stop_reason':None}
