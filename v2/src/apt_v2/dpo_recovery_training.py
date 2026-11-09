"""Contracts for one-pass recovery DPO and development recovery action evaluation."""
import math,random
from collections import Counter
from .tasks import make_tasks
from .dpo import pair_id,action_from_text
from .dpo_recovery import SCENARIOS,assess
from .dpo_training import input_id

def validate_training_pairs(pairs):
    allowed={(t.id,t.family_id) for t in make_tasks() if t.split=='train'}
    if len(pairs)!=169 or len({p['pair_id'] for p in pairs})!=169:raise ValueError('169 distinct reviewed pairs required')
    if any(p['split']!='train' or (p['task_id'],p['family_id']) not in allowed or p['pair_id']!=pair_id(p) for p in pairs):raise ValueError('train pair provenance differs')
    sources=Counter(p['rejected_source'] for p in pairs)
    if sources!={'frozen_sft_sample':44,'scripted_failure':84,'frozen_dpo_sample':41}:raise ValueError('reviewed source mix differs')
    for p in pairs:
        if p['rejected_source']!='frozen_sft_sample' and (p['scenario'] not in SCENARIOS or p['context_id']!=p['task_id']+':'+p['scenario']):raise ValueError('recovery scenario identity differs')
    return dict(sources)

def training_batches(count,batch_size,seed):
    if type(count) is not int or type(batch_size) is not int or min(count,batch_size)<=0:raise ValueError('positive integer data and batch sizes required')
    indices=list(range(count));random.Random(seed).shuffle(indices)
    return [indices[i:i+batch_size] for i in range(0,count,batch_size)]

def verify_reference(pairs,encoded,reference,adapter_sha256):
    if reference['baseline_adapter_sha256']!=adapter_sha256 or reference['pair_ids']!=[p['pair_id'] for p in pairs] or len(reference['scores'])!=len(encoded):raise ValueError('reference identity differs')
    for enc,ref in zip(encoded,reference['scores']):
        if ref['input_id']!=input_id(enc) or len(ref['chosen'])!=len(enc['chosen']) or len(ref['rejected'])!=len(enc['rejected']):raise ValueError('reference token binding differs')
        if any(not math.isfinite(v) or v>1e-6 for v in ref['chosen']+ref['rejected']):raise ValueError('finite reference log probabilities required')

def validate_generation(tokenizer,messages,sample,mode):
    ids=sample['response_token_ids'];raw=sample['raw_token_log_probs']
    if sample['mode']!=mode or not ids or raw is None or len(ids)!=len(raw):raise ValueError('recovery native token alignment differs')
    if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite native probabilities required')
    if sample['prompt_token_ids']!=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True):raise ValueError('recovery native prompt differs')
    if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=sample['text']:raise ValueError('recovery native decode differs')

def recovery_result(task,context,sample,tokenizer,mode):
    if task.split!='development':raise ValueError('development-only recovery evaluation required')
    validate_generation(tokenizer,context['messages'],sample,mode)
    row,reason=assess(task,context,sample['text'],allowed_split='development')
    action=action_from_text(sample['text'])
    return {'row':row,'reason':reason,'productive':reason is None,
        'strict_format':set(action)=={'tool','arguments'},
        'repeated_previous_action':action==context['records'][-1]['row']['action']}

def summarize_recovery(rows):
    return {'samples':len(rows),'productive':sum(r['result']['productive'] for r in rows),
        'strict_format':sum(r['result']['strict_format'] for r in rows),
        'tool_errors':sum(r['result']['row']['tool_error'] is not None for r in rows),
        'repeated_previous_action':sum(r['result']['repeated_previous_action'] for r in rows),
        'reasons':dict(Counter(r['result']['reason'] for r in rows if r['result']['reason']))}


def optimize_epoch(model,encoded,reference,optimizer,batch_size,seed,beta,max_grad_norm,on_step=None):
    import torch
    from .dpo_training import pair_objective
    if len(encoded)!=len(reference) or not encoded:raise ValueError('aligned nonempty recovery training data required')
    parameters=[p for p in model.parameters() if p.requires_grad];history=[];cursor=0
    model.train()
    for step,batch in enumerate(training_batches(len(encoded),batch_size,seed),1):
        optimizer.zero_grad(set_to_none=True);value=0.
        for i in batch:
            loss,_=pair_objective(model,encoded[i],reference[i],beta,True)
            if not torch.isfinite(loss):raise ValueError('nonfinite recovery DPO objective')
            (loss/len(batch)).backward();value+=loss.detach().item()/len(batch)
        norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
        if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
        optimizer.step();cursor+=len(batch)
        row={'step':step,'pairs_seen':cursor,'loss':value,'grad_norm':norm.item()};history.append(row)
        if on_step is not None:on_step(row)
    return history


def optimize_epochs(model,encoded,reference,optimizer,batch_size,seed,beta,max_grad_norm,epochs,on_step=None):
    if type(epochs) is not int or epochs<=0:raise ValueError('positive integer epoch count required')
    steps_per_epoch=len(training_batches(len(encoded),batch_size,seed));history=[]
    for epoch in range(epochs):
        def record(local):
            row={**local,'epoch':epoch+1,'epoch_step':local['step'],'epoch_pairs_seen':local['pairs_seen'],
                'step':epoch*steps_per_epoch+local['step'],'pairs_seen':epoch*len(encoded)+local['pairs_seen']}
            history.append(row)
            if on_step is not None:on_step(row)
        optimize_epoch(model,encoded,reference,optimizer,batch_size,seed+epoch,beta,max_grad_norm,record)
    return history

def verify_update_history(cfg,final,train,state=None):
    steps=cfg['max_steps'];exposures=cfg['pair_count']*cfg['epochs']
    if final['steps']!=steps or final['cursor']!=exposures or final['source_dpo_updates']!=11 or final['total_dpo_updates']!=11+steps or train['optimizer_start']!='fresh_adam' or train['optimizer_updates']!=steps or train['pair_exposures']!=exposures or train['frozen_parameter_gradients']!=0:raise ValueError('recovery update evidence differs')
    expected=[]
    for epoch in range(cfg['epochs']):
        batches=training_batches(cfg['pair_count'],cfg['gradient_accumulation'],cfg['seed']+epoch);cursor=epoch*cfg['pair_count']
        for epoch_step,batch in enumerate(batches,1):
            cursor+=len(batch);expected.append((len(expected)+1,cursor,epoch+1,epoch_step))
    if len(expected)!=steps or len(train['history'])!=steps:raise ValueError('complete epoch history required')
    for row,(step,cursor,epoch,epoch_step) in zip(train['history'],expected):
        if row['step']!=step or row['pairs_seen']!=cursor or (cfg['epochs']>1 and (row['epoch']!=epoch or row['epoch_step']!=epoch_step)):raise ValueError('epoch schedule differs')
        if not math.isfinite(row['loss']) or not math.isfinite(row['grad_norm']) or row['grad_norm']<=0:raise ValueError('finite update metrics required')
    if state is not None:
        if state['steps']!=steps or state['cursor']!=exposures or state['history']!=train['history'] or not state['optimizer']['state'] or any(int(v['step'])!=steps for v in state['optimizer']['state'].values()):raise ValueError('fresh Adam update count differs')
