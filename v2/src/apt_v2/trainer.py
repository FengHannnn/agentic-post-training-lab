"""Serial action-token PPO surrogate with turn-level verifiable credit."""
import torch
from .rewards import assign_turn_credit,training_samples

def response_scores(model,prompt,response,grad=False):
    device=next(model.parameters()).device
    ids=torch.tensor([prompt+response],device=device)
    with torch.set_grad_enabled(grad):
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
        positions=logits[0,len(prompt)-1:-1].float().log_softmax(-1)
        return positions.gather(-1,torch.tensor(response,device=device).unsqueeze(-1)).squeeze(-1)

def prepare_scores(model,reference,episodes):
    if any(p.requires_grad or p.grad is not None for p in reference.parameters()):
        raise ValueError('reference model must be frozen')
    model.eval();reference.eval()
    cache={}
    for i,episode in enumerate(episodes):
        for sample in training_samples(episode):
            args=sample['prompt_token_ids'],sample['response_token_ids']
            old=response_scores(model,*args,False)
            fresh=response_scores(model,*args,True)
            if not torch.allclose(old,fresh.detach(),atol=1e-5,rtol=0):
                raise ValueError('same-path old/new gate failed')
            cache[i,sample['turn']]={'prompt_token_ids':list(args[0]),'response_token_ids':list(args[1]),'old':old.cpu().detach(),
                'reference':response_scores(reference,*args,False).cpu().detach()}
    return cache

def update(model,optimizer,episodes,cache,beta=.02,clip=.2,tis_cap=2.,max_grad_norm=1.):
    if beta<0 or not 0<clip<1 or tis_cap<=0 or max_grad_norm<=0:raise ValueError('invalid optimizer settings')
    parameters=[p for name,p in model.named_parameters() if p.requires_grad]
    if not parameters or any('lora_' not in name for name,p in model.named_parameters() if p.requires_grad):
        raise ValueError('only LoRA parameters may be trainable')
    credits=assign_turn_credit(episodes);model.eval();optimizer.zero_grad(set_to_none=True)
    device=next(model.parameters()).device
    metrics={'loss':0.,'actions':0,'sampled_k3':0.,'clip_fraction':0.}
    for i,(episode,credit) in enumerate(zip(episodes,credits)):
        for sample,turn_credit in zip(training_samples(episode),credit['turns']):
            entry=cache[i,sample['turn']]
            if entry['prompt_token_ids']!=sample['prompt_token_ids'] or entry['response_token_ids']!=sample['response_token_ids']:
                raise ValueError('cached action identity differs')
            new=response_scores(model,sample['prompt_token_ids'],sample['response_token_ids'],True)
            old=entry['old'].to(device).detach();ref=entry['reference'].to(device).detach()
            q=torch.tensor(sample['behavior_token_log_probs'],device=device).detach()
            if old.shape!=new.shape or ref.shape!=new.shape or q.shape!=new.shape:
                raise ValueError('aligned action probabilities required')
            ratio=(new-old).exp()
            weights=(old-q).exp().clamp(max=tis_cap).detach()
            advantage=turn_credit['advantage']
            policy=-torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
            delta=ref.double()-new.double()
            k3=torch.expm1(delta)-delta
            loss=(weights*(policy+beta*k3.to(new.dtype))).mean()/len(episodes)
            if not torch.isfinite(loss):raise ValueError('nonfinite action objective')
            loss.backward()
            metrics['loss']+=loss.detach().item();metrics['actions']+=1
            metrics['sampled_k3']+=k3.detach().mean().item()
            metrics['clip_fraction']+=((ratio.detach()-1).abs()>clip).float().mean().item()
    for key in ('sampled_k3','clip_fraction'):metrics[key]/=metrics['actions']
    if metrics['sampled_k3']>.05 or metrics['clip_fraction']>.5:
        optimizer.zero_grad(set_to_none=True);return {**metrics,'updated':False}
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
    if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
    optimizer.step()
    return {**metrics,'updated':True,'grad_norm':norm.item()}
