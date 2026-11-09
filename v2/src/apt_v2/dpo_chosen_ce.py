"""Sequence-sum DPO plus an unweighted token-mean chosen CE anchor."""
import math
from .dpo import response_logps,dpo_loss
from .dpo_training import input_id
from .dpo_recovery_training import training_batches,verify_update_history

def pair_terms(model,encoded,reference,beta,grad=False):
    import torch
    if reference['input_id']!=input_id(encoded):raise ValueError('reference input binding differs')
    chosen=response_logps(model,encoded['prompt'],encoded['chosen'],grad);rejected=response_logps(model,encoded['prompt'],encoded['rejected'],grad)
    rc=torch.tensor(reference['chosen'],device=chosen.device);rr=torch.tensor(reference['rejected'],device=chosen.device)
    if rc.shape!=chosen.shape or rr.shape!=rejected.shape:raise ValueError('reference token alignment differs')
    return dpo_loss(chosen.sum(),rejected.sum(),rc.sum(),rr.sum(),beta),-chosen.mean()

def combine_loss(dpo,ce,weight,coefficient,batch_size):
    # Conditional keeps the zero-CE control identical to the original weighted path.
    return dpo*weight/batch_size if coefficient==0 else (dpo*weight+coefficient*ce)/batch_size

def optimize_chosen_ce(model,encoded,reference,weights,optimizer,batch_size,seed,beta,max_grad_norm,epochs,coefficient,on_step=None):
    import torch
    if not encoded or len(encoded)!=len(reference) or len(encoded)!=len(weights):raise ValueError('aligned nonempty preference data required')
    if any(not math.isfinite(w) or w<=0 for w in weights) or not math.isfinite(coefficient) or coefficient<0:raise ValueError('finite positive weights and nonnegative coefficient required')
    if type(epochs) is not int or epochs<=0:raise ValueError('positive epochs required')
    parameters=[p for p in model.parameters() if p.requires_grad];history=[];cursor=0;model.train()
    for epoch in range(epochs):
        epoch_cursor=0
        for epoch_step,batch in enumerate(training_batches(len(encoded),batch_size,seed+epoch),1):
            optimizer.zero_grad(set_to_none=True);value=dpo_value=ce_value=0.
            for i in batch:
                dpo,ce=pair_terms(model,encoded[i],reference[i],beta,True)
                loss=combine_loss(dpo,ce,weights[i],coefficient,len(batch))
                if not torch.isfinite(loss) or not torch.isfinite(ce):raise ValueError('finite DPO and chosen CE required')
                loss.backward();value+=loss.detach().item();dpo_value+=(dpo.detach()*weights[i]/len(batch)).item();ce_value+=(ce.detach()/len(batch)).item()
            norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
            if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
            optimizer.step();cursor+=len(batch);epoch_cursor+=len(batch)
            row={'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':epoch_step,'epoch_pairs_seen':epoch_cursor,'loss':value,'weighted_dpo_loss':dpo_value,'chosen_ce':ce_value,'chosen_ce_coefficient':coefficient,'grad_norm':norm.item()};history.append(row)
            if on_step is not None:on_step(row)
    return history

def verify_chosen_ce_updates(cfg,branch,final,train,state=None):
    verify_update_history(cfg,final,train,state);coefficient=cfg['training_variants'][branch]['chosen_ce_coefficient']
    expected_algorithm='hard_recovery_weighted_dpo' if coefficient==0 else 'weighted_dpo_chosen_ce'
    for obj in (final,train):
        if obj['branch']!=branch or obj['algorithm']!=expected_algorithm or obj['chosen_ce_coefficient']!=coefficient or obj['source_sft_updates']!=255 or obj['total_sft_updates']!=255 or obj['total_dpo_updates']!=269 or obj['epochs_completed']!=6 or obj['auxiliary_chosen_ce_updates']!=(258 if coefficient else 0):raise ValueError('paired DPO+CE ancestry differs')
    for row in train['history']:
        if row['chosen_ce_coefficient']!=coefficient or row['chosen_ce']<0 or not math.isfinite(row['chosen_ce']) or not math.isfinite(row['weighted_dpo_loss']) or abs(row['loss']-row['weighted_dpo_loss']-coefficient*row['chosen_ce'])>1e-5 or row['epoch_pairs_seen']!=row['pairs_seen']-(row['epoch']-1)*169:raise ValueError('DPO+CE objective history differs')
    if train['chosen_ce_pair_exposures']!=(1014 if coefficient else 0):raise ValueError('chosen CE exposure differs')

def numeric_equal(a,b):
    if isinstance(a,float) and isinstance(b,(float,int)):
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(numeric_equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(numeric_equal(x,y) for x,y in zip(a,b))
    return a==b
