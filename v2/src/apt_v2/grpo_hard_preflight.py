"""Frozen train-only, equal-budget paired contexts for hard-recovery signal diagnosis."""
import copy,json
from dataclasses import asdict
from .tasks import metric_sql,truth
from .environment import Environment
from .runtime import SYSTEM
from .dpo import action_from_text,corrective_action
from .grpo import initial_state,digest
from .grpo_credit import compute_credit,LOCAL
from .grpo_curriculum import build_context as original_context,training_tasks,TARGET_SCENARIOS

LEVELS=('simplified','compound')
TASK_IDS=('train-000','curriculum-train-000','train-001','curriculum-train-001')


def tasks():
    pool={t.id:t for t in training_tasks({'max_groups':48})}
    result=[pool[k] for k in TASK_IDS]
    if any(t.split!='train' or t.max_turns!=12 for t in result):raise ValueError('four frozen train tasks with total budget 12 required')
    return result


def context(task,scenario,allowed_split='train'):
    if allowed_split!='train' or task.split!='train':raise ValueError('training-only preflight contexts required')
    base,level=scenario.rsplit('__',1)
    if base not in TARGET_SCENARIOS or level not in LEVELS:raise ValueError('known paired preflight condition required')
    original=original_context(task,base,'train')
    if level=='compound':return original
    actions=copy.deepcopy([r['row']['action'] for r in original['records']])
    if base=='missing_alias_pair':
        prior=metric_sql(task,'prior_net_cents');indices=[i for i,a in enumerate(actions) if a.get('tool')=='query_sql' and a['arguments']['sql']==prior.rsplit(' AS ',1)[0]]
        if len(indices)!=1:raise ValueError('one prior-month missing-alias action required')
        actions[indices[0]]['arguments']['sql']=prior
    elif base=='schema_query_repeat':
        index=max(i for i,a in enumerate(actions) if a.get('tool')=='query_sql')
        actions[index]={'tool':'inspect_schema','arguments':{}}
    else:
        if actions[-1]['tool']!='check_result':raise ValueError('final failed check required')
        actions[-1]['arguments']['values']['change_cents']=truth(task)['change_cents']
    env=Environment(task);records=[];messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    try:
        for action in actions:
            text=json.dumps(action);row=env.step(action_from_text(text));records.append({'text':text,'row':row})
            messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if env.done or len(records)!=len(original['records']):raise ValueError('paired prefix length or unfinished state differs')
        chosen=corrective_action(env);chosen_row=env.step(chosen)
        return {**original,'context_id':task.id+':'+scenario,'scenario':scenario,'records':records,'messages':messages,
                'chosen':json.dumps(chosen),'chosen_observation':chosen_row['observation']}
    finally:env.close()


def plan(cfg):
    result=[]
    for ti,task in enumerate(tasks()):
        for si,base in enumerate(TARGET_SCENARIOS):
            levels=LEVELS if (ti+si)%2==0 else LEVELS[::-1]
            for level in levels:
                for ri in range(4):
                    result.append({'index':len(result),'task_id':task.id,'base_scenario':base,'level':level,
                        'scenario':base+'__'+level,'rollout_index':ri,'group_index':len(result)//4,
                        'seed':cfg['seed']+ti*1000000+si*10000+ri*100,'temperature':1})
    return result


def expert(task,scenario):
    env,_,_,_=initial_state(task,scenario,context_builder=context)
    try:
        prefix=len(env.records);initial_verified=sorted(env.verified);first=corrective_action(env)
        while not env.done:env.step(corrective_action(env))
        if not env.success:raise ValueError('paired expert continuation must succeed')
        return {'task_id':task.id,'scenario':scenario,'prefix_turns':prefix,'remaining_turns':12-prefix,
                'initial_verified_metrics':initial_verified,'first_expert_action':first,
                'continuation_turns':len(env.records)-prefix,'total_turns':len(env.records),'passed':True}
    finally:env.close()


def frozen_inputs(cfg):
    proofs=[expert(t,s+'__'+level) for t in tasks() for s in TARGET_SCENARIOS for level in LEVELS]
    for i in range(0,len(proofs),2):
        if proofs[i]['prefix_turns']!=proofs[i+1]['prefix_turns']:raise ValueError('equal continuation budget required')
    return {'tasks.json':[asdict(t) for t in tasks()],'plan.json':plan(cfg),'expert_budget_proof.json':proofs,
        'contexts.json':[{'task_id':t.id,'scenario':s+'__'+level,'context':context(t,s+'__'+level)} for t in tasks() for s in TARGET_SCENARIOS for level in LEVELS]}


def summarize(episodes):
    if len(episodes)!=96:raise ValueError('all 96 preflight episodes required')
    rows=[];credits=[]
    for gi in range(24):
        es=episodes[4*gi:4*gi+4];credit=compute_credit(es,method=LOCAL);credits.append(credit)
        for e in es:
            new=sum(len(r['new_verified_metrics']) for r in e['turns'])
            initial={m for item in e['start']['prefix'] for m in item['row']['new_verified_metrics']}
            missing=3-len(initial);seen_failures=set();repeated=0
            def failed(r):return bool(r['tool_error']) or r['observation'].get('passed') is False or (r['action'].get('tool')=='query_sql' and not r['observation'].get('verified_metrics'))
            for item in e['start']['prefix']:
                if failed(item['row']):seen_failures.add(digest(item['row']['action']))
            for r in e['turns']:
                if failed(r):
                    key=digest(r['action']);repeated+=key in seen_failures;seen_failures.add(key)
            rows.append({'task_id':e['task_id'],'base_scenario':e['base_scenario'],'level':e['level'],
                'rollout_index':e['rollout_index'],'success':e['success'],'new_verified_metrics':new,
                'has_progress':new>0,'initial_verified_metrics':len(initial),'initially_missing_metrics':missing,
                'remaining_metric_completion':new/missing,'repeated_failed_actions':repeated,'actions':len(e['turns']),'tool_errors':e['tool_errors'],
                'failed_checks':sum(r['action'].get('tool')=='check_result' and r['observation'].get('passed') is False for r in e['turns']),
                'prefix_turns':e['prefix_turns'],'remaining_budget':12-e['prefix_turns']})
    def stats(base,level):
        rs=[r for r in rows if r['base_scenario']==base and r['level']==level]
        cs=[credits[i] for i in range(24) if episodes[4*i]['base_scenario']==base and episodes[4*i]['level']==level]
        return {'episodes':len(rs),'successes':sum(r['success'] for r in rs),'progress_episodes':sum(r['has_progress'] for r in rs),
                'new_verified_metrics':sum(r['new_verified_metrics'] for r in rs),'eligible_groups':sum(c['eligible'] for c in cs),'groups':len(cs),
                'nonzero_advantage_actions':sum(a!=0 for c in cs for v in c['turn_advantages'] for a in v),
                'groups_with_local_progress':sum(any(v>0 for row in c['local_progress_rewards'] for v in row) for c in cs),
                'outcome_variable_groups':sum(not c['outcome_zero_variance'] for c in cs),
                'mean_remaining_metric_completion':sum(r['remaining_metric_completion'] for r in rs)/len(rs),
                'repeated_failed_actions':sum(r['repeated_failed_actions'] for r in rs),
                'actions':sum(r['actions'] for r in rs),'tool_errors':sum(r['tool_errors'] for r in rs),'failed_checks':sum(r['failed_checks'] for r in rs)}
    by={s:{level:stats(s,level) for level in LEVELS} for s in TARGET_SCENARIOS};paired={}
    for s in TARGET_SCENARIOS:
        levels={level:{(r['task_id'],r['rollout_index']):r for r in rows if r['base_scenario']==s and r['level']==level} for level in LEVELS}
        a,b=levels['compound'],levels['simplified'];assert a.keys()==b.keys()
        paired[s]={metric:{'improved':sum(not a[k][metric] and b[k][metric] for k in a),
                   'regressed':sum(a[k][metric] and not b[k][metric] for k in a),
                   'delta':sum(int(b[k][metric])-int(a[k][metric]) for k in a)} for metric in ('success','has_progress')}
    return {'episodes':96,'groups':24,'new_training_updates':0,'credit_method':LOCAL,'by_scenario':by,'paired_simplified_vs_compound':paired,'episode_rows':rows}
