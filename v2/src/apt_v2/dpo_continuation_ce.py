"""Train-only expert continuations with a fixed CE action exposure budget."""
import json,math,random
from collections import Counter
from .tasks import make_tasks
from .environment import Environment
from .dpo_recovery import SCENARIOS,build_context
from .dpo import corrective_action,action_from_text,TEMPORARY_ERROR
from .dpo_investigation import digest
from .dpo_recovery_training import training_batches,verify_update_history
from .dpo_chosen_ce import pair_terms
from .sft import action_loss

def build_continuations():
    examples=[];episodes=[]
    for task in make_tasks():
        if task.split!='train':continue
        for scenario in SCENARIOS:
            context=build_context(task,scenario,allowed_split='train');env=Environment(task);messages=[dict(m) for m in context['messages']];rows=[]
            try:
                for item in context['records']:
                    if env.step(action_from_text(item['text']))!=item['row']:raise ValueError('expert training prefix replay differs')
                if env.done:raise ValueError('unfinished expert training context required')
                while not env.done:
                    action=corrective_action(env);text=json.dumps(action);row=env.step(action)
                    if row['tool_error'] not in (None,TEMPORARY_ERROR):raise ValueError('expert training action failed')
                    examples.append({'example_id':context['context_id']+':continuation:'+str(len(rows)),'task_id':task.id,'family_id':task.family_id,'split':'train','scenario':scenario,'context_id':context['context_id'],'continuation_index':len(rows),'expected_tool':action['tool'],'messages':[dict(m) for m in messages],'response':text})
                    rows.append({'text':text,'row':row});messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
                if not env.success:raise ValueError('successful expert training continuation required')
                episodes.append({'task_id':task.id,'family_id':task.family_id,'split':'train','context_id':context['context_id'],'scenario':scenario,'prefix':context['records'],'turns':rows,'success':env.success,'artifact':rows[-1]['row']['observation']['artifact']})
            finally:env.close()
    if len(episodes)!=84 or len({e['example_id'] for e in examples})!=len(examples) or not any(e['expected_tool']=='submit' for e in examples):raise ValueError('complete distinct train-only continuations required')
    return examples,episodes

def ce_schedule(count,budget,seed):
    if any(type(x) is not int for x in (count,budget,seed)) or count<=0 or budget<=0:raise ValueError('positive integer CE pool and budget required')
    indices=[];cycle=0
    while len(indices)<budget:
        order=list(range(count));random.Random(seed+cycle).shuffle(order);indices.extend(order);cycle+=1
    return indices[:budget]

def coverage(examples,schedule):
    if any(type(i) is not int or not 0<=i<len(examples) for i in schedule):raise ValueError('CE schedule index differs')
    counts=Counter(schedule)
    return {'pool_actions':len(examples),'pool_contexts':len({e['context_id'] for e in examples}),'pool_tasks':len({e['task_id'] for e in examples}),'pool_tool_counts':dict(Counter(e['expected_tool'] for e in examples)),'sampled_actions':len(schedule),'sampled_unique_actions':len(counts),'minimum_exposures_per_pool_action':min(counts.get(i,0) for i in range(len(examples))),'maximum_exposures_per_pool_action':max(counts.values()),'sampled_tool_counts':dict(Counter(examples[i]['expected_tool'] for i in schedule)),'sampled_task_counts':dict(Counter(examples[i]['task_id'] for i in schedule)),'test_or_development_training_examples':sum(e['split']!='train' for e in examples)}

def token_budget(encoded,ce_encoded,schedule,epochs):
    return {'DPO_pair_exposures':len(encoded)*epochs,'CE_action_exposures':len(schedule),'DPO_chosen_response_tokens':epochs*sum(len(e['chosen']) for e in encoded),'DPO_rejected_response_tokens':epochs*sum(len(e['rejected']) for e in encoded),'DPO_input_tokens':epochs*sum(2*len(e['prompt'])+len(e['chosen'])+len(e['rejected']) for e in encoded),'CE_response_tokens':sum(ce_encoded[i]['response_length'] for i in schedule),'CE_extra_forward_input_tokens':sum(len(ce_encoded[i]['input_ids']) for i in schedule),'CE_extra_forwards':len(schedule)}

def optimize_continuation_ce(model,encoded,reference,weights,ce_encoded,schedule,optimizer,batch_size,seed,beta,max_grad_norm,epochs,coefficient,on_step=None):
    import torch
    if not encoded or len(encoded)!=len(reference) or len(encoded)!=len(weights) or len(schedule)!=len(encoded)*epochs or not ce_encoded:raise ValueError('aligned fixed DPO and CE budgets required')
    if coefficient!=1 or any(not math.isfinite(w) or w<=0 for w in weights) or any(type(i) is not int or not 0<=i<len(ce_encoded) for i in schedule):raise ValueError('fixed CE coefficient and valid schedule required')
    parameters=[p for p in model.parameters() if p.requires_grad];history=[];cursor=0;model.train()
    for epoch in range(epochs):
        epoch_cursor=0
        for epoch_step,batch in enumerate(training_batches(len(encoded),batch_size,seed+epoch),1):
            optimizer.zero_grad(set_to_none=True);dpo_value=ce_value=0.;indices=schedule[cursor:cursor+len(batch)]
            for i,j in zip(batch,indices):
                dpo,_=pair_terms(model,encoded[i],reference[i],beta,True);weighted=dpo*weights[i]/len(batch)
                if not torch.isfinite(weighted):raise ValueError('finite weighted DPO required')
                weighted.backward();dpo_value+=weighted.detach().item()
                ce=action_loss(model,ce_encoded[j]);auxiliary=ce*coefficient/len(batch)
                if not torch.isfinite(auxiliary):raise ValueError('finite continuation CE required')
                auxiliary.backward();ce_value+=(ce.detach()/len(batch)).item()
            norm=torch.nn.utils.clip_grad_norm_(parameters,max_grad_norm,error_if_nonfinite=True)
            if norm.item()<=0:raise ValueError('nonzero LoRA gradient required')
            optimizer.step();cursor+=len(batch);epoch_cursor+=len(batch)
            row={'step':len(history)+1,'pairs_seen':cursor,'epoch':epoch+1,'epoch_step':epoch_step,'epoch_pairs_seen':epoch_cursor,'loss':dpo_value+coefficient*ce_value,'weighted_dpo_loss':dpo_value,'chosen_ce':ce_value,'chosen_ce_coefficient':coefficient,'ce_indices':indices,'ce_actions_seen':cursor,'CE_response_tokens':sum(ce_encoded[j]['response_length'] for j in indices),'CE_extra_forward_input_tokens':sum(len(ce_encoded[j]['input_ids']) for j in indices),'grad_norm':norm.item()};history.append(row)
            if on_step is not None:on_step(row)
    return history

def verify_continuation_updates(cfg,final,train,state=None):
    verify_update_history(cfg,final,train,state)
    for obj in (final,train):
        if obj['algorithm']!='weighted_dpo_continuation_ce' or obj['branch']!='continuation_ce' or obj['chosen_ce_coefficient']!=1 or obj['auxiliary_chosen_ce_updates']!=258 or obj['source_sft_updates']!=255 or obj['total_sft_updates']!=255 or obj['total_dpo_updates']!=269 or obj['epochs_completed']!=6:raise ValueError('continuation CE ancestry differs')
    if train['chosen_ce_pair_exposures']!=1014 or train['ce_actions_seen']!=1014:raise ValueError('continuation CE exposure differs')
    expected=ce_schedule(train['ce_coverage']['pool_actions'],1014,cfg['ce_seed']);cursor=0
    for r in train['history']:
        n=r['pairs_seen']-cursor
        if r['ce_indices']!=expected[cursor:cursor+n] or r['ce_actions_seen']!=r['pairs_seen'] or r['chosen_ce_coefficient']!=1 or not math.isfinite(r['chosen_ce']) or r['chosen_ce']<0 or not math.isfinite(r['weighted_dpo_loss']) or abs(r['loss']-r['weighted_dpo_loss']-r['chosen_ce'])>1e-5 or r['epoch_pairs_seen']!=r['pairs_seen']-(r['epoch']-1)*169:raise ValueError('continuation CE objective or exposure schedule differs')
        cursor=r['pairs_seen']
