"""Frozen C4 holdout instances, paired summaries and practical acceptance."""
import math
import random
from dataclasses import asdict
from .tasks import Task,make_tasks,connect
from .dpo_recovery import SCENARIOS
from .grpo import digest,initial_state,run_continuation,replay
from .dpo import corrective_action
from .runtime import Generation
import json

LABELS=('weighted_sft','outcome_rl16','fine_rl17')
ADAPTERS={'weighted_sft':'f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f',
          'outcome_rl16':'6599dcbb8286a9850c1b175ddbb95b5507e3f6dd734755fe9eb833778dd609e6',
          'fine_rl17':'1990f06d2eb8e3a1777273051bf2bce078c8bff67206533b88dd65780e5959c6'}
VERSIONS={'weighted_sft':0,'outcome_rl16':16,'fine_rl17':17}


def confirmation_tasks(cfg=None):
    months=(1,2,4,5,7,8,10,11,3,6,9,12)
    count=8 if cfg is None else cfg['task_count']
    if count not in (8,12):raise ValueError('frozen confirmation task count required')
    regions=('East','North','Central','West','East','South','Central','North','South','West','East','Central')
    return [Task(f'c4-confirmation-{i:03}','test',f'c4-family-{91000+i}',91000+i,
                 f'2027-{month:02}',('2026-12' if month==1 else f'2027-{month-1:02}'),
                 regions[i],f'C4-Segment-{91000+i}','transient_query' if i%2 else 'none',12)
            for i,month in enumerate(months[:count])]


def independence_report(cfg=None):
    from .period_closing import closing_tasks
    tasks=confirmation_tasks(cfg);historical=make_tasks()+closing_tasks()
    for key in ('id','family_id','seed'):
        values=[getattr(t,key) for t in tasks]
        if len(set(values))!=len(tasks) or set(values)&{getattr(t,key) for t in historical}:
            raise ValueError('new confirmation task identities overlap')
    fingerprints=[]
    for task in tasks:
        db=connect(task)
        try: fingerprints.append(digest([db.execute('SELECT * FROM '+table+' ORDER BY 1').fetchall() for table in ('orders','refunds')]))
        finally: db.close()
    if len(set(fingerprints))!=len(tasks):raise ValueError('distinct confirmation databases required')
    return {'passed':True,'task_count':len(tasks),'excluded_historical_task_count':len(historical),
            'disjoint_fields':['id','family_id','seed'],'database_sha256':fingerprints,
            'scope':'new instances and family identities of the existing monthly reconciliation business',
            'new_business_template':False,'tasks':[asdict(t) for t in tasks]}


def confirmation_plan(cfg):
    result=[]
    for ti,t in enumerate(confirmation_tasks(cfg)):
        for si,scenario in enumerate((None,*SCENARIOS)):
            for mode in ('greedy','sample'):
                count=1 if mode=='greedy' else (4 if scenario is None else 3)
                for ri in range(count):
                    result.append({'task_id':t.id,'scenario':scenario,'kind':'recovery' if scenario else 'complete',
                        'mode':mode,'rollout_index':ri,'temperature':0 if mode=='greedy' else .8,
                        'seed':cfg['confirmation_seed']+ti*1000000+si*10000+(5000 if mode=='sample' else 0)+ri*100})
    return result


def key(row):
    return (row['task_id'],row['scenario'],row.get('evaluation_mode',row.get('mode')),row['rollout_index'])


def summarize(episodes,cfg=None):
    def stats(rows):return {'episodes':len(rows),'successes':sum(e['success'] for e in rows),
        'actions':sum(len(e['turns']) for e in rows),
        'response_tokens':sum(len(r['generation']['response_token_ids']) for e in rows for r in e['turns']),
        'tool_errors':sum(e['tool_errors'] for e in rows)}
    result={}
    for mode in ('greedy','sample'):
        result[mode]={}
        for kind in ('complete','recovery'):
            rows=[e for e in episodes if e['evaluation_mode']==mode and bool(e['scenario'])==(kind=='recovery')]
            result[mode][kind]={**stats(rows),'by_task':{t.id:stats([e for e in rows if e['task_id']==t.id]) for t in confirmation_tasks(cfg)},
                'by_scenario':{s:stats([e for e in rows if e['scenario']==s]) for s in SCENARIOS} if kind=='recovery' else {}}
    return result


def paired(before,after,bootstrap_seed=2026110711,repetitions=10000,cfg=None):
    a={key(e):e for e in before};b={key(e):e for e in after}
    if len(a)!=len(before) or len(b)!=len(after) or a.keys()!=b.keys():raise ValueError('unique complete paired keys required')
    result={}
    def stats(keys):
        n=len(keys);improved=sum(not a[k]['success'] and b[k]['success'] for k in keys);regressed=sum(a[k]['success'] and not b[k]['success'] for k in keys)
        return {'episodes':n,'both_success':sum(a[k]['success'] and b[k]['success'] for k in keys),
                'both_failure':sum(not a[k]['success'] and not b[k]['success'] for k in keys),
                'improved':improved,'regressed':regressed,'delta_successes':improved-regressed,
                'delta_rate':(improved-regressed)/n if n else 0.}
    for mode in ('greedy','sample'):
        result[mode]={}
        for kind in ('complete','recovery'):
            keys=[k for k in a if k[2]==mode and bool(k[1])==(kind=='recovery')]
            by_task={t.id:stats([k for k in keys if k[0]==t.id]) for t in confirmation_tasks(cfg)}
            rates=[v['delta_rate'] for v in by_task.values()];rng=random.Random(bootstrap_seed)
            draws=sorted(math.fsum(rng.choice(rates) for _ in rates)/len(rates) for _ in range(repetitions))
            result[mode][kind]={**stats(keys),'by_task':by_task,
                'task_cluster_bootstrap_95pct':[draws[int(.025*(repetitions-1))],draws[int(.975*(repetitions-1))]],
                'bootstrap_seed':bootstrap_seed,'bootstrap_repetitions':repetitions,'interval_role':'descriptive'}
    return result


def acceptance(baseline,candidate,cfg=None):
    minimum=4 if cfg is None else cfg['minimum_recovery_gain']
    denominator=len(confirmation_tasks(cfg))*21
    full=baseline['sample']['complete'];primary='recovery' if full['successes']==full['episodes'] else 'complete'
    delta=lambda mode,kind:candidate[mode][kind]['successes']-baseline[mode][kind]['successes']
    checks={f'sample_recovery_gain_at_least_{minimum}':delta('sample','recovery')>=minimum,
            'sample_complete_no_regression':delta('sample','complete')>=0,
            'greedy_complete_no_regression':delta('greedy','complete')>=0,
            'greedy_recovery_no_regression':delta('greedy','recovery')>=0,
            'primary_success_improves':delta('sample',primary)>0}
    return {'passed':all(checks.values()),'primary_metric':primary,'checks':checks,
            'sample_recovery_delta':delta('sample','recovery'),'sample_complete_delta':delta('sample','complete'),
            'minimum_sample_recovery_delta':minimum,'sample_recovery_denominator':denominator,'new_training_updates':0}


def expert_confirmation(task,scenario):
    env,_,_,_=initial_state(task,scenario)
    def policy(messages):
        action=corrective_action(env);env.step(action);return Generation(json.dumps(action))
    try:
        e=run_continuation(task,scenario,policy,{'temperature':0},0,0,0)
        replay(task,e)
        if not e['success'] or e['total_turns']>12:raise ValueError('holdout expert must complete within unchanged budget')
        return {'task_id':task.id,'scenario':scenario,'passed':True,'prefix_turns':e['prefix_turns'],
                'generated_actions':len(e['turns']),'total_turns':e['total_turns']}
    finally:env.close()
