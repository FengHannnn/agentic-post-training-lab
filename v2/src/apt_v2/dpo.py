"""Verified same-context action preferences and sequence-sum DPO objective."""
import json,math,hashlib
from .environment import Environment,METRICS
from .runtime import SYSTEM,parse_action
from .tasks import metric_sql

TEMPORARY_ERROR='temporary query service error; retry the query'

def action_from_text(text):
    try:return parse_action(text)
    except (ValueError,TypeError):return {'invalid_action':text}

def corrective_action(env):
    if not env.policy_read:return {'tool':'read_policy','arguments':{}}
    if not env.schema_read:return {'tool':'inspect_schema','arguments':{}}
    for metric in ('current_net_cents','prior_net_cents'):
        if metric not in env.verified:return {'tool':'query_sql','arguments':{'sql':metric_sql(env.task,metric)}}
    evidence={};values={}
    for eid,metrics in env.evidence.items():
        for metric,value in metrics.items():evidence[metric]=eid;values[metric]=value
    if 'change_cents' not in env.verified:
        return {'tool':'calculate_change','arguments':{'current_evidence':evidence['current_net_cents'],
            'prior_evidence':evidence['prior_net_cents']}}
    checked=any(r['action'].get('tool')=='check_result' and r['observation'].get('passed') for r in env.records)
    return {'tool':'submit' if checked else 'check_result','arguments':{
        'values':{m:values[m] for m in METRICS},'evidence':{m:evidence[m] for m in METRICS},'policy_id':'revenue-v1'}}

def failure_reason(row):
    if row['tool_error']==TEMPORARY_ERROR:return None
    if row['tool_error']:return 'tool_error'
    observation=row['observation']
    if row['action'].get('tool')=='query_sql' and not observation.get('verified_metrics'):return 'sql_unverified'
    if row['action'].get('tool') in ('check_result','submit') and observation.get('passed') is False:return 'failed_check'
    return None

def prefix_environment(task,records,reward_mode):
    env=Environment(task,reward_mode)
    try:
        for row in records:
            actual=env.step(action_from_text(row['generation']['text']))
            for key in ('turn','action','observation','reward','done','success','tool_error','new_verified_metrics'):
                if actual[key]!=row[key]:raise ValueError('prefix replay differs: '+key)
        return env
    except Exception:
        env.close();raise

def pair_id(pair):
    fields={k:pair[k] for k in ('messages','chosen','rejected')}
    return hashlib.sha256(json.dumps(fields,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def verified_pairs(task,episode,tokenizer,max_length=2048,max_pairs=2):
    if task.split!='train' or episode['split']!='train' or episode['task_id']!=task.id:
        raise ValueError('same train task required')
    env=Environment(task,episode['reward_mode']);messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    pairs=[]
    try:
        for index,row in enumerate(episode['turns']):
            sample=row['generation'];ids=sample['response_token_ids'];raw=sample['raw_token_log_probs']
            if sample['mode']!='dpo_sample' or not ids or raw is None or len(ids)!=len(raw):raise ValueError('DPO sample token alignment required')
            if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite raw log probabilities required')
            prompt=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
            if sample['prompt_token_ids']!=prompt:raise ValueError('sample prompt differs')
            if tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=sample['text']:raise ValueError('sample decode differs')
            chosen_action=corrective_action(env)
            actual=env.step(action_from_text(sample['text']))
            for key in ('turn','action','observation','reward','done','success','tool_error','process_reward','outcome_reward','new_verified_metrics','termination'):
                if actual[key]!=row[key]:raise ValueError('sample replay differs: '+key)
            reason=failure_reason(actual)
            if reason and len(pairs)<max_pairs and sample['finish_reason']=='stop' and ids[-1]==tokenizer.eos_token_id:
                chosen=json.dumps(chosen_action)
                fork=prefix_environment(task,episode['turns'][:index],episode['reward_mode'])
                try:
                    before=(fork.policy_read,fork.schema_read,len(fork.verified))
                    chosen_row=fork.step(chosen_action)
                    after=(fork.policy_read,fork.schema_read,len(fork.verified))
                    meaningful=before!=after or chosen_row['observation'].get('passed') is True
                    good=chosen_row['tool_error'] is None and meaningful
                finally:fork.close()
                if good:
                    pair={'task_id':task.id,'family_id':task.family_id,'split':'train','turn':row['turn'],
                        'rollout_id':episode['rollout_id'],'messages':[dict(m) for m in messages],
                        'chosen':chosen,'rejected':sample['text'],'reason':reason,
                        'chosen_source':'scripted_counterfactual','rejected_source':'frozen_sft_sample',
                        'chosen_observation':chosen_row['observation'],'rejected_observation':actual['observation'],
                        'native_prompt_token_ids':list(prompt),'native_rejected_token_ids':list(ids)}
                    try:encoded=encode_pair(tokenizer,pair,max_length)
                    except ValueError as exc:
                        if 'context exceeds max_length' not in str(exc):raise
                    else:
                        pair['pair_id']=pair_id(pair);pair['chosen_tokens']=len(encoded['chosen']);pair['rejected_tokens']=len(encoded['rejected'])
                        pairs.append(pair)
            messages += [{'role':'assistant','content':sample['text']},
                {'role':'user','content':json.dumps({'observation':row['observation']})}]
        if not env.done or env.success!=episode['success'] or env.tool_errors!=episode['tool_errors']:
            raise ValueError('sample episode completion differs')
        if (env.success and env.progress_after_error)!=episode['recovered']:raise ValueError('sample recovery differs')
        return pairs
    finally:env.close()

def encode_pair(tokenizer,pair,max_length):
    from .sft import encode_example
    chosen=encode_example(tokenizer,{'messages':pair['messages'],'response':pair['chosen']},max_length)
    prompt=chosen['input_ids'][:chosen['prompt_length']]
    rejected=pair['native_rejected_token_ids']
    if prompt!=pair['native_prompt_token_ids']:raise ValueError('pair contexts differ')
    if not rejected or rejected[-1]!=tokenizer.eos_token_id or tokenizer.eos_token_id in rejected[:-1]:raise ValueError('rejected native EOS required')
    if tokenizer.decode(rejected,skip_special_tokens=True,clean_up_tokenization_spaces=False)!=pair['rejected']:raise ValueError('rejected decode differs')
    if len(prompt)+len(rejected)>max_length:raise ValueError('context exceeds max_length')
    if pair['chosen']==pair['rejected']:raise ValueError('distinct preference responses required')
    return {'prompt':prompt,'chosen':chosen['input_ids'][chosen['prompt_length']:],'rejected':list(rejected)}

def response_logps(model,prompt,response,grad=False):
    import torch
    device=next(model.parameters()).device
    ids=torch.tensor([prompt+response],device=device);n=len(response)
    if not prompt or not response:raise ValueError('nonempty prompt and response required')
    with torch.set_grad_enabled(grad):
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=n+1).logits[0,-n-1:-1].float()
        target=torch.tensor(response,device=device)
        return logits.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)

def dpo_loss(chosen,rejected,reference_chosen,reference_rejected,beta=.1):
    import torch
    if not math.isfinite(beta) or beta<=0:raise ValueError('positive finite beta required')
    values=(chosen,rejected,reference_chosen,reference_rejected)
    if any(v.ndim!=0 or not torch.isfinite(v) for v in values):raise ValueError('finite sequence-sum log probabilities required')
    if reference_chosen.requires_grad or reference_rejected.requires_grad:raise ValueError('detached frozen reference scores required')
    margin=(chosen-rejected)-(reference_chosen-reference_rejected)
    return -torch.nn.functional.logsigmoid(beta*margin)


def balanced_pairs(candidates,max_per_task=4):
    from collections import Counter
    if type(max_per_task) is not int or max_per_task<=0:raise ValueError('positive per-task pair cap required')
    result=[];seen=set();counts=Counter()
    for pair in candidates:
        if pair['pair_id']!=pair_id(pair) or pair['split']!='train':raise ValueError('pair identity or split differs')
        if pair['pair_id'] in seen or counts[pair['task_id']]>=max_per_task:continue
        seen.add(pair['pair_id']);counts[pair['task_id']]+=1;result.append(pair)
    return result
