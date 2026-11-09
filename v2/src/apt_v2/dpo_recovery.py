"""Train-only controlled recovery contexts and verified action preferences."""
import json,math
from .environment import Environment
from .runtime import SYSTEM
from .tasks import metric_sql
from .dpo import corrective_action,action_from_text,failure_reason,pair_id,encode_pair,TEMPORARY_ERROR
from .sft import encode_example

SCENARIOS=('format_repair','unknown_tool_repair','sql_unverified_requery','repeat_evidence_error','stale_evidence_repair','repeat_sql_error','failed_check_repair')

def build_context(task,scenario,allowed_split='train'):
    if allowed_split not in ('train','development','test') or task.split!=allowed_split or scenario not in SCENARIOS:raise ValueError('known recovery scenario in explicitly allowed train/development/test split required')
    env=Environment(task);records=[];messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    def step(text):
        if not isinstance(text,str):text=json.dumps(text)
        row=env.step(action_from_text(text));records.append({'text':text,'row':row})
        messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if row['tool_error']==TEMPORARY_ERROR:return step(text)
        return row
    def query(metric):return step({'tool':'query_sql','arguments':{'sql':metric_sql(task,metric)}})
    bad_sql={'tool':'query_sql','arguments':{'sql':'SELECT SUM(gross_cents) AS current_net_cents FROM orders'}}
    bad_evidence={'tool':'calculate_change','arguments':{'current_evidence':'missing-current','prior_evidence':'missing-prior'}}
    try:
        step({'tool':'read_policy','arguments':{}});step({'tool':'inspect_schema','arguments':{}})
        if scenario in ('sql_unverified_requery','repeat_evidence_error'):
            bad=step(bad_sql);query('prior_net_cents')
            assert not bad['observation'].get('verified_metrics')
            if scenario=='repeat_evidence_error':
                bad_evidence={'tool':'calculate_change','arguments':{'current_evidence':bad['observation']['evidence_id'],
                    'prior_evidence':next(eid for eid,metrics in env.evidence.items() if 'prior_net_cents' in metrics)}}
                step(bad_evidence);step(bad_evidence)
            rejected=json.dumps(bad_evidence)
        elif scenario=='repeat_sql_error':
            invalid={'tool':'query_sql','arguments':{'sql':'SELECT current_net_cents FROM metrics'}}
            step(invalid);step(invalid);rejected=json.dumps(invalid)
        else:
            query('current_net_cents');query('prior_net_cents')
            if scenario=='format_repair':
                text='Here is the action: '+json.dumps(corrective_action(env));step(text);rejected=text
            elif scenario=='unknown_tool_repair':
                invalid={'tool':'calculate_net_revenue','arguments':{}};step(invalid);rejected=json.dumps(invalid)
            elif scenario=='stale_evidence_repair':
                step(bad_evidence);step(bad_evidence);rejected=json.dumps(bad_evidence)
            else:
                step(corrective_action(env));invalid=corrective_action(env)
                invalid['arguments']['values']['change_cents']+=1
                step(invalid);rejected=json.dumps(invalid)
        if env.done:raise ValueError('recovery prefix exhausted task budget')
        chosen_action=corrective_action(env);before=(env.policy_read,env.schema_read,len(env.verified))
        chosen_row=env.step(chosen_action);after=(env.policy_read,env.schema_read,len(env.verified))
        if chosen_row['tool_error'] is not None or not (before!=after or chosen_row['observation'].get('passed') is True):
            raise ValueError('productive expert recovery required')
        return {'context_id':task.id+':'+scenario,'task_id':task.id,'family_id':task.family_id,'split':task.split,'scenario':scenario,
            'prefix_source':'scripted_error_injection','records':records,'messages':messages,'chosen':json.dumps(chosen_action),
            'chosen_observation':chosen_row['observation'],'scripted_rejected':rejected}
    finally:env.close()

def assess(task,context,text,allowed_split='train'):
    if build_context(task,context['scenario'],allowed_split)!=context:raise ValueError('recovery context replay differs')
    env=Environment(task)
    try:
        for item in context['records']:
            if env.step(action_from_text(item['text']))!=item['row']:raise ValueError('prefix row differs')
        before=(env.policy_read,env.schema_read,len(env.verified));row=env.step(action_from_text(text))
        after=(env.policy_read,env.schema_read,len(env.verified))
        reason=failure_reason(row)
        if reason is None and before==after and row['observation'].get('passed') is not True:reason='no_progress'
        if row['tool_error']==TEMPORARY_ERROR:reason=None
        return row,reason
    finally:env.close()

def encode_recovery_pair(tokenizer,pair,max_length):
    if 'context_id' not in pair:return encode_pair(tokenizer,pair,max_length)
    chosen=encode_example(tokenizer,{'messages':pair['messages'],'response':pair['chosen']},max_length)
    prompt=chosen['input_ids'][:chosen['prompt_length']];ids=pair['rejected_token_ids']
    if prompt!=pair['context_token_ids']:raise ValueError('recovery pair contexts differ')
    if not ids or ids[-1]!=tokenizer.eos_token_id or tokenizer.eos_token_id in ids[:-1]:raise ValueError('complete rejected EOS required')
    if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=pair['rejected']:raise ValueError('rejected decode differs')
    if len(prompt)+len(ids)>max_length:raise ValueError('context exceeds max_length')
    if pair['chosen']==pair['rejected']:raise ValueError('distinct responses required')
    if pair['rejected_source']=='frozen_dpo_sample':
        raw=pair['rejected_raw_token_log_probs']
        if pair['rejected_token_origin']!='native_sample' or raw is None or len(raw)!=len(ids) or any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('native rejected probability alignment required')
    elif pair['rejected_source']=='scripted_failure':
        encoded=encode_example(tokenizer,{'messages':pair['messages'],'response':pair['rejected']},max_length)
        if pair['rejected_token_origin']!='chat_template' or pair['rejected_raw_token_log_probs'] is not None or encoded['input_ids'][encoded['prompt_length']:]!=ids:raise ValueError('scripted token provenance differs')
    else:raise ValueError('known rejected source required')
    return {'prompt':prompt,'chosen':chosen['input_ids'][chosen['prompt_length']:],'rejected':list(ids)}

def make_pair(task,context,tokenizer,max_length,sample=None):
    if sample is None:
        text=context['scripted_rejected'];encoded=encode_example(tokenizer,{'messages':context['messages'],'response':text},max_length)
        ids=encoded['input_ids'][encoded['prompt_length']:];raw=None;source='scripted_failure';origin='chat_template'
    else:
        text=sample['text'];ids=sample['response_token_ids'];raw=sample['raw_token_log_probs'];source='frozen_dpo_sample';origin='native_sample'
        prompt=tokenizer.apply_chat_template(context['messages'],tokenize=True,add_generation_prompt=True)
        if sample['mode']!='dpo_sample' or sample['prompt_token_ids']!=prompt:raise ValueError('native recovery prompt differs')
        if not ids or raw is None or len(ids)!=len(raw) or any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('native recovery probabilities differ')
        if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=text:raise ValueError('native recovery decode differs')
        if sample['finish_reason']!='stop' or ids[-1]!=tokenizer.eos_token_id:return None
    row,reason=assess(task,context,text)
    if reason is None:return None
    pair={k:context[k] for k in ('context_id','task_id','family_id','split','scenario','messages','prefix_source','chosen','chosen_observation')}
    pair.update(turn=len(context['records']),rejected=text,reason=reason,chosen_source='scripted_counterfactual',rejected_source=source,
        rejected_token_origin=origin,rejected_observation=row['observation'],rejected_token_ids=list(ids),rejected_raw_token_log_probs=raw,
        context_token_ids=list(tokenizer.apply_chat_template(context['messages'],tokenize=True,add_generation_prompt=True)))
    if sample is not None:pair['sample_id']=sample['sample_id']
    try:encoded=encode_recovery_pair(tokenizer,pair,max_length)
    except ValueError as exc:
        if 'context exceeds max_length' in str(exc):return None
        raise
    pair.update(pair_id=pair_id(pair),chosen_tokens=len(encoded['chosen']),rejected_tokens=len(encoded['rejected']))
    return pair

def select_pairs(anchors,scripted,native):
    result=[];seen=set();native_contexts=set()
    for pair in anchors+scripted+native:
        if pair['split']!='train' or pair['pair_id']!=pair_id(pair):raise ValueError('train pair identity differs')
        if pair['pair_id'] in seen:continue
        if pair.get('rejected_source')=='frozen_dpo_sample':
            if pair['context_id'] in native_contexts:continue
            native_contexts.add(pair['context_id'])
        result.append(pair);seen.add(pair['pair_id'])
    return result
