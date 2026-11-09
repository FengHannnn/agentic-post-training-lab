"""Scenario-weighted chosen-only SFT for recovery actions."""
import math
from collections import Counter
from .chosen_sft import chosen_examples
from .dpo_recovery_training import validate_training_pairs,training_batches
from .sql_sft import backward_batch

HARD_SCENARIOS=('sql_unverified_requery','repeat_evidence_error','repeat_sql_error','failed_check_repair')

def recovery_weights(pairs,hard_weight=4.):
    validate_training_pairs(pairs)
    if isinstance(hard_weight,bool) or not math.isfinite(hard_weight) or hard_weight<=0:raise ValueError('positive finite recovery weight required')
    raw=[float(hard_weight) if p.get('scenario') in HARD_SCENARIOS else 1. for p in pairs]
    mean=sum(raw)/len(raw);weights=[w/mean for w in raw]
    return weights,{'hard_scenarios':list(HARD_SCENARIOS),'hard_examples':sum(p.get('scenario') in HARD_SCENARIOS for p in pairs),
        'other_examples':sum(p.get('scenario') not in HARD_SCENARIOS for p in pairs),'raw_hard_weight':float(hard_weight),
        'normalization_mean':mean,'normalized_mean':sum(weights)/len(weights),
        'hard_weight_share':sum(w for p,w in zip(pairs,weights) if p.get('scenario') in HARD_SCENARIOS)/sum(weights),
        'counts_by_scenario':dict(Counter(p.get('scenario','original_sql') for p in pairs))}

def optimize_weighted_sft(model,encoded,weights,optimizer,batch_size,seed,max_grad_norm,epochs,on_step=None):
    import torch
    if not encoded or len(encoded)!=len(weights) or any(not math.isfinite(w) or w<=0 for w in weights):raise ValueError('aligned nonempty samples and positive finite weights required')
    if type(epochs) is not int or epochs<=0:raise ValueError('positive integer epochs required')
    parameters=[p for p in model.parameters() if p.requires_grad];history=[];cursor=0;model.train()
    for epoch in range(epochs):
        epoch_cursor=0
        for epoch_step,indices in enumerate(training_batches(len(encoded),batch_size,seed+epoch),1):
            optimizer.zero_grad(set_to_none=True)
            value=backward_batch(model,encoded,weights,indices)
            norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
            if norm.item()<=0:raise ValueError('nonzero recovery gradient required')
            optimizer.step();cursor+=len(indices);epoch_cursor+=len(indices)
            row={'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':epoch_step,
                'epoch_pairs_seen':epoch_cursor,'loss':value,'grad_norm':norm.item()};history.append(row)
            if on_step is not None:on_step(row)
    return history

def verify_weighted_updates(cfg,final,train,state=None):
    steps=cfg['max_steps'];exposures=cfg['pair_count']*cfg['epochs']
    if final['algorithm']!='hard_recovery_weighted_sft' or final['steps']!=steps or final['cursor']!=exposures or final['epochs_completed']!=cfg['epochs']:
        raise ValueError('weighted recovery checkpoint differs')
    if final['source_sft_updates']!=cfg['source_sft_updates'] or final['source_dpo_updates']!=cfg['source_dpo_updates'] or final['total_sft_updates']!=cfg['source_sft_updates']+steps or final['total_dpo_updates']!=cfg['source_dpo_updates']:
        raise ValueError('weighted recovery ancestry differs')
    if train['algorithm']!='hard_recovery_weighted_sft' or train['optimizer_start']!='fresh_adam' or train['optimizer_updates']!=steps or train['pair_exposures']!=exposures or train['frozen_parameter_gradients']!=0:
        raise ValueError('weighted recovery optimizer evidence differs')
    expected=[]
    for epoch in range(cfg['epochs']):
        cursor=epoch*cfg['pair_count']
        for ei,batch in enumerate(training_batches(cfg['pair_count'],cfg['gradient_accumulation'],cfg['seed']+epoch),1):
            cursor+=len(batch);expected.append((len(expected)+1,cursor,epoch+1,ei,cursor-epoch*cfg['pair_count']))
    if len(expected)!=steps or len(train['history'])!=steps:raise ValueError('complete weighted history required')
    for row,entry in zip(train['history'],expected):
        if (row['step'],row['pairs_seen'],row['epoch'],row['epoch_step'],row['epoch_pairs_seen'])!=entry:raise ValueError('weighted schedule differs')
        if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) or row['grad_norm']<=0:raise ValueError('finite weighted updates required')
    if state is not None:
        if state['steps']!=steps or state['cursor']!=exposures or state['history']!=train['history'] or not state['optimizer']['state'] or any(int(s['step'])!=steps for s in state['optimizer']['state'].values()):
            raise ValueError('weighted Adam step differs')
