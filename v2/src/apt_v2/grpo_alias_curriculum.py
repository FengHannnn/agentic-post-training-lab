"""Missing-alias curriculum with fixed local credit, paired phases and fresh development."""
import copy,json
from dataclasses import asdict
from .tasks import Task,metric_sql,make_tasks
from .grpo_credit import LOCAL
from .dpo_recovery import SCENARIOS as STANDARD

SCHEMA='alias_curriculum_v1'
BRANCHES=('control','curriculum')
LEVELS=('simplified','compound')
TARGET_SCENARIOS=tuple('missing_alias_pair__'+level for level in LEVELS)


def training_tasks(cfg):
    from .grpo_curriculum import training_tasks as old_tasks
    return old_tasks({'max_groups':48})[:12]


def development_tasks():
    from .grpo_curriculum import month_pair
    result=[]
    for i,month in enumerate((1,4,8,12)):
        current,prior=month_pair(2032+i%2,month)
        result.append(Task(f'alias-development-{i:03}','development',f'alias-family-{81000+i}',81000+i,
            current,prior,('Central','East','West','South')[i],f'AliasDevSegment-{81000+i}',
            'transient_query' if i%2 else 'none'))
    return result


def assert_isolation(cfg):
    from .grpo_curriculum import development_tasks as old_dev
    from .grpo_fine_study import development_tasks as fine_dev
    from .grpo_confirmation import confirmation_tasks
    train=training_tasks(cfg);dev=development_tasks()
    excluded=[t for t in make_tasks() if t.split!='train']+old_dev()+fine_dev()+confirmation_tasks({'protocol_version':2,'task_count':12})
    for field in ('id','family_id','seed','segment'):
        a,b,c=({getattr(t,field) for t in ts} for ts in (train,dev,excluded))
        if len(a)!=12 or len(b)!=4 or a&b or a&c or b&c:raise ValueError('alias study isolation differs: '+field)
    return {'passed':True,'train_tasks':12,'development_tasks':4,'excluded_instances':len(excluded)}


def validate_config(cfg):
    if (cfg.get('curriculum_schema')!=SCHEMA or cfg['study_phase']!='alias_curriculum'
            or cfg['curriculum_arm'] not in BRANCHES or cfg['algorithm']!='fine_grained_grpo'
            or cfg['credit_method']!=LOCAL or cfg['learning_rate']!=1e-6 or cfg['k']!=4
            or cfg['max_groups']!=48 or cfg['max_updates']!=48 or cfg['group_evaluation_cursors']!=[48]
            or cfg['max_wall_seconds']!=21600 or cfg['formal_gpu_budget_seconds']!=21600
            or cfg['packaging_reserve_seconds']!=180 or cfg['minimum_actual_updates']!=0
            or cfg['recovery_scenarios']!=list(STANDARD)+list(TARGET_SCENARIOS)
            or cfg.get('closing_evaluation')!={'enabled':False}
            or cfg['targeted_evaluation']!={'enabled':True,'seed':2026110929,'samples_per_context':3}):
        raise ValueError('frozen alias curriculum contract differs')
    assert_isolation(cfg)


def start_spec(cfg,index):
    if type(index) is not int or not 0<=index<=48:raise ValueError('bounded alias group cursor required')
    tasks=training_tasks(cfg);pair=index//2;phase=min(index//24+1,2)
    task=tasks[pair%12]
    if index%2:
        level='simplified' if cfg['curriculum_arm']=='curriculum' and index<24 else 'compound'
        scenario='missing_alias_pair__'+level;role='target'
    else:
        # 12 complete plus 12 standard recovery maintenance groups; repeat task pairing in phase 2.
        complete=(pair%2==0) == (index<24)
        recovery_index=(pair%12)//2+(0 if index<24 else 6)
        scenario=None if complete else STANDARD[recovery_index%len(STANDARD)]
        role='maintenance'
    return {'index':index,'task_id':task.id,'scenario':scenario,'kind':'recovery' if scenario else 'complete',
            'seed':cfg['seed']+index*100000,'phase':phase,'training_role':role}


def context(task,scenario,allowed_split='train'):
    from .grpo_curriculum import build_context as original
    from .environment import Environment
    from .runtime import SYSTEM
    from .dpo import corrective_action,action_from_text
    if scenario not in TARGET_SCENARIOS:return original(task,scenario,allowed_split)
    if task.split!=allowed_split or allowed_split not in ('train','development'):raise ValueError('alias train/development split differs')
    base=original(task,'missing_alias_pair',allowed_split)
    if scenario.endswith('__compound'):
        return {**base,'context_id':task.id+':'+scenario,'scenario':scenario}
    actions=copy.deepcopy([r['row']['action'] for r in base['records']]);prior=metric_sql(task,'prior_net_cents')
    indices=[i for i,a in enumerate(actions) if a.get('tool')=='query_sql' and a['arguments']['sql']==prior.rsplit(' AS ',1)[0]]
    if len(indices)!=1:raise ValueError('one prior alias edit required')
    actions[indices[0]]['arguments']['sql']=prior
    env=Environment(task);records=[];messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    try:
        for action in actions:
            text=json.dumps(action);row=env.step(action_from_text(text));records.append({'text':text,'row':row})
            messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if env.done or len(records)!=len(base['records']) or set(env.verified)!={'prior_net_cents'}:
            raise ValueError('paired prefix budget or certified prior differs')
        chosen=corrective_action(env);row=env.step(chosen)
        return {**base,'context_id':task.id+':'+scenario,'scenario':scenario,'records':records,'messages':messages,
                'chosen':json.dumps(chosen),'chosen_observation':row['observation']}
    finally:env.close()


def targeted_plan(cfg,midpoint=False):
    return [{'task_id':t.id,'scenario':s,'kind':'targeted_recovery','mode':mode,'rollout_index':ri,
             'seed':cfg['targeted_evaluation']['seed']+ti*100000+si*10000+ri*100+(80000000 if midpoint else 0),
             'temperature':0 if mode=='greedy' else cfg['recovery_eval']['temperature']}
            for ti,t in enumerate(development_tasks()) for si,s in enumerate(TARGET_SCENARIOS)
            for mode in (('sample',) if midpoint else ('greedy','sample'))
            for ri in range(2 if midpoint else (1 if mode=='greedy' else 3))]


def summarize_targeted(episodes):
    def stats(rows):
        return {'episodes':len(rows),'successes':sum(e['success'] for e in rows),
                'progress_episodes':sum(any(r['new_verified_metrics'] for r in e['turns']) for e in rows),
                'new_verified_metrics':sum(len(r['new_verified_metrics']) for e in rows for r in e['turns']),
                'actions':sum(len(e['turns']) for e in rows),
                'response_tokens':sum(len(r['generation']['response_token_ids']) for e in rows for r in e['turns'])}
    return {mode:{**stats([e for e in episodes if e['evaluation_mode']==mode]),'by_scenario':{
        s:stats([e for e in episodes if e['evaluation_mode']==mode and e['scenario']==s]) for s in TARGET_SCENARIOS}}
        for mode in ('greedy','sample')}


def expert_proof(task,scenario):
    from .grpo import initial_state
    from .dpo import corrective_action
    env,_,_,_=initial_state(task,scenario,context_builder=context)
    try:
        prefix=len(env.records);initial=sorted(env.verified)
        while not env.done:env.step(corrective_action(env))
        if not env.success:raise ValueError('alias study expert must finish within total budget')
        return {'task_id':task.id,'scenario':scenario,'prefix_turns':prefix,'initial_verified_metrics':initial,
                'continuation_turns':len(env.records)-prefix,'total_turns':len(env.records),'passed':True}
    finally:env.close()


def frozen_inputs(cfg):
    tasks=training_tasks(cfg);dev=development_tasks();proofs=[expert_proof(t,s) for t in tasks+dev for s in (None,*STANDARD,*TARGET_SCENARIOS)]
    return {'curriculum_training_tasks.json':[asdict(t) for t in tasks],
            'curriculum_group_plan.json':[start_spec(cfg,i) for i in range(49)],
            'curriculum_isolation.json':assert_isolation(cfg),'curriculum_expert_proof.json':proofs,
            'targeted_evaluation_plan.json':targeted_plan(cfg),'midpoint_plan.json':targeted_plan(cfg,True),
            'targeted_recovery_contexts.json':[context(t,s,'development') for t in dev for s in TARGET_SCENARIOS],
            'alias_training_contexts.json':[context(t,s) for t in tasks for s in TARGET_SCENARIOS],
            'alias_method.json':{'credit_method':LOCAL,'learning_rate':1e-6,'maintenance_fraction':.5,
                'phase_switch_group_cursor':24,'phase_switch_is_fixed':True,'reset_optimizer_at_switch':False,
                'extra_sft_updates':0,'extra_dpo_updates':0,'reference':'W','midpoint_used_for_selection':False}}


def exposure(cfg,count):
    plan=[start_spec(cfg,i) for i in range(count)]
    return {str(phase):{role:sum(p['phase']==phase and p['training_role']==role for p in plan)
                       for role in ('maintenance','target')} for phase in (1,2)}


def choose_candidate(summaries):
    from .grpo_fine_study import choose_candidate as old
    result=old({'weighted_sft':summaries['weighted_sft'],'legacy':summaries['control'],'local':summaries['curriculum']})
    rename=lambda b:{'legacy':'control','local':'curriculum',None:None}[b]
    return {**{k:v for k,v in result.items() if k not in ('checks','selected_candidate','local_minus_legacy_primary')},
            'checks':{rename(b):c for b,c in result['checks'].items()},'selected_candidate':rename(result['selected_candidate']),
            'curriculum_minus_control_primary':result['local_minus_legacy_primary']}
