"""Chosen-only SFT control preserving the reviewed preference-pair sampling weights."""
import hashlib,json,math
from .dpo_recovery_training import validate_training_pairs,training_batches
from .sft import encode_example,action_loss

def chosen_examples(pairs):
    validate_training_pairs(pairs)
    return [{'task_id':p['task_id'],'family_id':p['family_id'],'split':'train','source_pair_id':p['pair_id'],
        'messages':[dict(m) for m in p['messages']],'response':p['chosen']} for p in pairs]

def data_sha256(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def encode_chosen(tokenizer,examples,max_length):
    return [encode_example(tokenizer,e,max_length) for e in examples]

def optimize_sft(model,encoded,optimizer,batch_size,seed,max_grad_norm,epochs,on_step=None):
    import torch
    if not encoded or type(epochs) is not int or epochs<=0:raise ValueError('nonempty data and positive epochs required')
    parameters=[p for p in model.parameters() if p.requires_grad];history=[];cursor=0;model.train()
    for epoch in range(epochs):
        epoch_cursor=0
        for epoch_step,batch in enumerate(training_batches(len(encoded),batch_size,seed+epoch),1):
            optimizer.zero_grad(set_to_none=True);value=0.
            for i in batch:
                loss=action_loss(model,encoded[i])
                if not torch.isfinite(loss):raise ValueError('finite chosen-only loss required')
                (loss/len(batch)).backward();value+=loss.detach().item()/len(batch)
            norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
            if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
            optimizer.step();cursor+=len(batch);epoch_cursor+=len(batch)
            row={'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':epoch_step,'epoch_pairs_seen':epoch_cursor,
                'loss':value,'grad_norm':norm.item()};history.append(row)
            if on_step is not None:on_step(row)
    return history

def verify_sft_updates(cfg,final,train,state=None):
    steps=cfg['max_steps'];exposures=cfg['pair_count']*cfg['epochs']
    if final['algorithm']!='chosen_only_sft' or final['steps']!=steps or final['cursor']!=exposures or final['source_dpo_updates']!=11 or final['sft_updates']!=129 or final['epochs_completed']!=3 or train['algorithm']!='chosen_only_sft' or train['optimizer_start']!='fresh_adam' or train['optimizer_updates']!=steps or train['pair_exposures']!=exposures or train['frozen_parameter_gradients']!=0:raise ValueError('chosen-only update evidence differs')
    expected=[]
    for epoch in range(cfg['epochs']):
        cursor=epoch*cfg['pair_count']
        for epoch_step,batch in enumerate(training_batches(cfg['pair_count'],cfg['gradient_accumulation'],cfg['seed']+epoch),1):
            cursor+=len(batch);expected.append((len(expected)+1,cursor,epoch+1,epoch_step))
    if len(train['history'])!=steps or len(expected)!=steps:raise ValueError('complete SFT epoch history required')
    for row,(step,cursor,epoch,epoch_step) in zip(train['history'],expected):
        if (row['step'],row['pairs_seen'],row['epoch'],row['epoch_step'])!=(step,cursor,epoch,epoch_step):raise ValueError('SFT epoch schedule differs')
        if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) or row['grad_norm']<=0:raise ValueError('finite SFT update metrics required')
    if state is not None:
        if state['steps']!=steps or state['cursor']!=exposures or state['history']!=train['history'] or not state['optimizer']['state'] or any(int(s['step'])!=steps for s in state['optimizer']['state'].values()):raise ValueError('SFT Adam update count differs')
