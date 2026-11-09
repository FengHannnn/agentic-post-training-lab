"""Bounded uniform DPO segments and predeclared development closeout gate."""
import math
from .dpo_recovery_training import training_batches
from .dpo_training import pair_objective

def optimize_segment(model,encoded,reference,optimizer,batch_size,seed,beta,max_grad_norm,target_step,history=None,on_step=None):
    import torch
    history=[dict(r) for r in (history or [])]
    batches=training_batches(len(encoded),batch_size,seed)
    if not encoded or len(encoded)!=len(reference) or type(target_step) is not int or not len(history)<target_step<=len(batches):raise ValueError('aligned data and increasing one-pass segment target required')
    if not math.isfinite(beta) or beta<=0:raise ValueError('positive beta required')
    params=[p for p in model.parameters() if p.requires_grad];model.train()
    for j in range(len(history),target_step):
        batch=batches[j];optimizer.zero_grad(set_to_none=True);value=0.
        for i in batch:
            loss,_=pair_objective(model,encoded[i],reference[i],beta,True)
            if not torch.isfinite(loss):raise ValueError('nonfinite short DPO loss')
            (loss/len(batch)).backward();value+=loss.detach().item()/len(batch)
        norm=torch.nn.utils.clip_grad_norm_(params,max_grad_norm,error_if_nonfinite=True)
        if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
        optimizer.step()
        row={'step':j+1,'pairs_seen':sum(len(b) for b in batches[:j+1]),'batch_indices':list(batch),'loss':value,'grad_norm':norm.item()}
        history.append(row)
        if on_step:on_step(row)
    return history

def verify_history(cfg,complete,train,state=None):
    step=complete['steps'];history=train['history'];batches=training_batches(cfg['pair_count'],cfg['gradient_accumulation'],cfg['seed'])
    if step not in cfg['checkpoint_steps'] or len(history)!=step or train['optimizer_updates']!=step:raise ValueError('short DPO checkpoint update count differs')
    cursor=0
    for j,row in enumerate(history):
        cursor+=len(batches[j])
        if row['step']!=j+1 or row['pairs_seen']!=cursor or row['batch_indices']!=batches[j]:raise ValueError('single shuffled pass schedule differs')
        if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) or row['grad_norm']<=0:raise ValueError('finite short DPO update required')
    for obj in (complete,train):
        if obj['algorithm']!=cfg['algorithm'] or obj['source_sft_updates']!=513 or obj['source_dpo_updates']!=11 or obj['total_sft_updates']!=513 or obj['total_dpo_updates']!=11+step:raise ValueError('weighted SFT initialization ancestry differs')
    if complete['cursor']!=cursor or train['pair_exposures']!=cursor or complete['epochs_completed']!=0 or train['epochs_completed']!=0:raise ValueError('partial single pass exposure differs')
    if train['optimizer_start']!='fresh_adam_at_step_zero_then_exact_resume' or train['frozen_parameter_gradients']!=0 or train['adapter_tensors_changed']<=0:raise ValueError('LoRA update evidence differs')
    if state is not None:
        if state['steps']!=step or state['history']!=history or state['cursor']!=cursor or not state['optimizer']['state'] or any(int(v['step'])!=step for v in state['optimizer']['state'].values()):raise ValueError('resumed Adam state differs')

def baseline_ready(summary,cfg):
    rule=cfg['selection_rule']
    return summary['sample']['complete']['successes']==rule['baseline_required_sample_complete_successes'] and summary['greedy']['complete']['successes']==rule['baseline_required_greedy_complete_successes']

def candidate_decision(base,candidate,cfg):
    r=cfg['selection_rule'];checks={};differences={}
    for mode in ('sample','greedy'):
        for kind in ('complete','recovery'):
            a=base[mode][kind];b=candidate[mode][kind]
            if a['episodes']!=b['episodes']:raise ValueError('paired gate denominator differs')
            delta=b['successes']-a['successes'];key=mode+'.'+kind
            differences[key]={'baseline':a['successes'],'candidate':b['successes'],'episodes':a['episodes'],'delta':delta,'percentage_points':100*delta/a['episodes']}
            limit=r['maximum_complete_success_drop' if kind=='complete' else 'maximum_recovery_success_drop'] if mode=='sample' else 0
            checks[key]=delta>=-limit
    checks['sample.complete_absolute_floor']=candidate['sample']['complete']['successes']>=r['minimum_sample_complete_successes']
    passed=all(checks.values());severe=candidate['sample']['complete']['successes']<=base['sample']['complete']['successes']-r['severe_complete_success_drop']
    return {'passed':passed,'severe_regression_stop':severe,'checks':checks,'differences':differences,'rule':'earliest_passing_checkpoint','statistical_scope':'fixed development operational acceptance rule'}

def select_candidate(records,summaries,cfg):
    # A passed checkpoint stops the runner; audit rejects evidence from later targets.
    candidates=[b for b in summaries if b!='weighted_sft'];passing=[b for b in candidates if records[b]['passed']]
    if passing:return min(passing,key=lambda b:int(b[-4:]))
    if not candidates:return None
    return max(candidates,key=lambda b:(summaries[b]['sample']['complete']['successes'],summaries[b]['sample']['recovery']['successes'],summaries[b]['greedy']['complete']['successes'],summaries[b]['greedy']['recovery']['successes'],-int(b[-4:])))
