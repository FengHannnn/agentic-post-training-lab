"""Sequence-sum DPO update with a token-bound, detached SFT reference cache."""
import hashlib,json
from .dpo import response_logps,dpo_loss

def input_id(encoded):
    return hashlib.sha256(json.dumps(encoded,sort_keys=True).encode()).hexdigest()

def reference_scores(model,encoded):
    return {'input_id':input_id(encoded),
        'chosen':response_logps(model,encoded['prompt'],encoded['chosen']).cpu().tolist(),
        'rejected':response_logps(model,encoded['prompt'],encoded['rejected']).cpu().tolist()}

def pair_objective(model,encoded,reference,beta,grad=False):
    import torch
    if reference['input_id']!=input_id(encoded):raise ValueError('reference input binding differs')
    chosen=response_logps(model,encoded['prompt'],encoded['chosen'],grad)
    rejected=response_logps(model,encoded['prompt'],encoded['rejected'],grad)
    rc=torch.tensor(reference['chosen'],device=chosen.device);rr=torch.tensor(reference['rejected'],device=chosen.device)
    if rc.shape!=chosen.shape or rr.shape!=rejected.shape:raise ValueError('reference token alignment differs')
    loss=dpo_loss(chosen.sum(),rejected.sum(),rc.sum(),rr.sum(),beta)
    metrics={'loss':loss.detach().item(),'relative_margin':((chosen.sum()-rejected.sum())-(rc.sum()-rr.sum())).detach().item(),
        'chosen_mean_logp':chosen.detach().mean().item(),'rejected_mean_logp':rejected.detach().mean().item()}
    return loss,metrics
