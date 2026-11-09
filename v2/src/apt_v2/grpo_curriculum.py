"""Frozen outcome-only curriculum, fresh development tasks and targeted diagnostics."""
import json
from dataclasses import asdict
from .tasks import Task, make_tasks, metric_sql
from .dpo_recovery import SCENARIOS as ORIGINAL_SCENARIOS, build_context as original_context
from .environment import Environment
from .runtime import SYSTEM
from .dpo import corrective_action, action_from_text, TEMPORARY_ERROR

SCHEMA = 'outcome_curriculum_v1'
TARGET_SCENARIOS = ('missing_alias_pair', 'schema_query_repeat', 'requery_stale_change')
SCENARIOS = (*ORIGINAL_SCENARIOS, *TARGET_SCENARIOS)
# Repeat the high-frequency SQL/evidence failures while covering every original scenario.
RECOVERY_CYCLE = ('missing_alias_pair', 'sql_unverified_requery', 'requery_stale_change',
                  'failed_check_repair', 'schema_query_repeat', 'repeat_evidence_error',
                  'format_repair', 'unknown_tool_repair', 'stale_evidence_repair', 'repeat_sql_error')


def enabled(cfg):
    return cfg.get('curriculum_schema') in (SCHEMA,'fine_credit_ablation_v1')


def month_pair(year, month):
    return f'{year}-{month:02}', f'{year if month > 1 else year-1}-{month-1 if month > 1 else 12:02}'


def training_tasks(cfg):
    old = [t for t in make_tasks() if t.split == 'train']
    fresh = []
    for i in range(12):
        current, prior = month_pair(2027+i%2, i+1)
        fresh.append(Task(f'curriculum-train-{i:03}', 'train', f'curriculum-family-{61000+i}',
                          61000+i, current, prior, ('East','North','Central','West','South')[i%5],
                          f'CurriculumSegment-{61000+i}', 'transient_query' if i%2 else 'none'))
    interleaved = [t for pair in zip(old, fresh) for t in pair]
    return interleaved[:cfg['max_groups']//2]


def development_tasks(cfg=None):
    if cfg and cfg.get('curriculum_schema')=='fine_credit_ablation_v1':
        from .grpo_fine_study import development_tasks as fine_tasks
        return fine_tasks()
    result = []
    for i, month in enumerate((1, 5, 9, 12)):
        current, prior = month_pair(2028+i%2, month)
        result.append(Task(f'curriculum-development-{i:03}', 'development', f'curriculum-family-{71000+i}',
                           71000+i, current, prior, ('Central','East','West','South')[i],
                           f'CurriculumDevSegment-{71000+i}', 'transient_query' if i%2 else 'none'))
    return result


def validate_config(cfg):
    if (not enabled(cfg) or cfg['algorithm'] != 'outcome_grpo' or cfg.get('credit_method') != 'outcome'
            or cfg['k'] != 4 or cfg['max_groups'] not in (24,48) or cfg['max_updates'] != cfg['max_groups']
            or cfg['group_evaluation_cursors'] != [cfg['max_groups']]
            or cfg['max_wall_seconds'] not in (14400,21600,25200)
            or cfg['formal_gpu_budget_seconds'] != cfg['max_wall_seconds']
            or (cfg['max_wall_seconds'] == 14400 and cfg['max_groups'] != 24)
            or (cfg['max_wall_seconds'] > 14400 and cfg['max_groups'] != 48)
            or cfg['learning_rate'] not in (1e-6,1e-5)
            or cfg['recovery_scenarios'] != list(SCENARIOS)
            or cfg.get('closing_evaluation',{}).get('enabled')
            or cfg.get('targeted_evaluation') != {'enabled':True,'seed':2026110719,'samples_per_context':1}):
        raise ValueError('frozen outcome curriculum contract differs')
    assert_isolation(cfg)


def assert_isolation(cfg):
    from .grpo_confirmation import confirmation_tasks
    train, dev = training_tasks(cfg), development_tasks(cfg)
    historic = [t for t in make_tasks() if t.split != 'train']
    if cfg.get('curriculum_schema')=='fine_credit_ablation_v1':historic+=development_tasks()
    c4 = confirmation_tasks({'protocol_version':2,'task_count':12})
    def values(tasks, field): return {getattr(t,field) for t in tasks}
    for field in ('id','family_id','seed','segment'):
        a,b,c = values(train,field),values(dev,field),values(historic+c4,field)
        if len(a)!=len(train) or len(b)!=len(dev) or a&b or a&c or b&c:
            raise ValueError('training/development/confirmation task isolation differs: '+field)
    return {'passed':True,'train_tasks':len(train),'development_tasks':len(dev),
            'c4_tasks_excluded':len(c4),'historic_development_test_excluded':len(historic)}


def start_spec(cfg, index):
    if type(index) is not int or not 0 <= index <= cfg['max_groups']:
        raise ValueError('bounded curriculum group cursor required')
    tasks = training_tasks(cfg)
    ti = (index//2)%len(tasks)
    # Rotate the second cycle so new failure types occur on both old and fresh families.
    scenario_index=(ti+ti//len(RECOVERY_CYCLE))%len(RECOVERY_CYCLE)
    scenario = RECOVERY_CYCLE[scenario_index] if index%2 else None
    return {'index':index,'task_id':tasks[ti].id,'scenario':scenario,
            'kind':'recovery' if scenario else 'complete','seed':cfg['seed']+index*100000}


def build_context(task, scenario, allowed_split='train'):
    if scenario in ORIGINAL_SCENARIOS:
        return original_context(task,scenario,allowed_split)
    if task.split != allowed_split or allowed_split not in ('train','development') or scenario not in TARGET_SCENARIOS:
        raise ValueError('known curriculum train/development recovery scenario required')
    env = Environment(task)
    records = []
    messages = [{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    def step(action):
        text = action if isinstance(action,str) else json.dumps(action)
        row = env.step(action_from_text(text))
        records.append({'text':text,'row':row})
        messages.extend([{'role':'assistant','content':text},
                         {'role':'user','content':json.dumps({'observation':row['observation']})}])
        if row['tool_error'] == TEMPORARY_ERROR: return step(action)
        return row
    def query(metric, alias=True):
        sql=metric_sql(task,metric)
        return step({'tool':'query_sql','arguments':{'sql':sql if alias else sql.rsplit(' AS ',1)[0]}})
    bad_evidence={'tool':'calculate_change','arguments':{'current_evidence':'missing-current','prior_evidence':'missing-prior'}}
    try:
        step({'tool':'read_policy','arguments':{}})
        step({'tool':'inspect_schema','arguments':{}})
        if scenario == 'missing_alias_pair':
            for metric in ('current_net_cents','prior_net_cents'):
                row=query(metric,False)
                if row['observation'].get('verified_metrics'):raise ValueError('missing aliases must leave metrics unverified')
            step(bad_evidence);rejected=bad_evidence
        elif scenario == 'schema_query_repeat':
            invalid={'tool':'query_sql','arguments':{'sql':'SELECT SUM(net_revenue_cents) AS current_net_cents FROM monthly_revenue'}}
            for _ in range(2):
                row=step(invalid)
                if not row['tool_error']:raise ValueError('schema injection must produce a real SQL error')
            step(bad_evidence);rejected=invalid
        else:
            query('current_net_cents',False);query('prior_net_cents');step(bad_evidence)
            current=query('current_net_cents')['observation']['evidence_id']
            evidence={};values={}
            for eid,metrics in env.evidence.items():
                for metric,value in metrics.items():evidence[metric]=eid;values[metric]=value
            rejected={'tool':'check_result','arguments':{'values':{**values,'change_cents':values['current_net_cents']-values['prior_net_cents']+1},
                      'evidence':{**evidence,'change_cents':current},'policy_id':'revenue-v1'}}
            row=step(rejected)
            if row['observation'].get('passed') is not False or 'change_cents' in env.verified:
                raise ValueError('stale change injection must require recalculation')
        if env.done:raise ValueError('curriculum prefix exhausted task budget')
        action=corrective_action(env)
        row=env.step(action)
        if row['tool_error'] or not (row['new_verified_metrics'] or row['observation'].get('passed')):
            raise ValueError('productive curriculum expert recovery required')
        return {'context_id':task.id+':'+scenario,'task_id':task.id,'family_id':task.family_id,
                'split':task.split,'scenario':scenario,'prefix_source':'scripted_error_injection',
                'records':records,'messages':messages,'chosen':json.dumps(action),
                'chosen_observation':row['observation'],'scripted_rejected':json.dumps(rejected)}
    finally:env.close()


def targeted_plan(cfg):
    if not enabled(cfg):return []
    return [{'task_id':t.id,'scenario':s,'kind':'targeted_recovery','mode':mode,'rollout_index':0,
             'seed':cfg['targeted_evaluation']['seed']+ti*100000+si*1000,
             'temperature':0 if mode=='greedy' else cfg['recovery_eval']['temperature']}
            for ti,t in enumerate(development_tasks(cfg)) for si,s in enumerate(TARGET_SCENARIOS)
            for mode in ('greedy','sample')]


def expert_proof(task, scenario):
    from .grpo import initial_state
    env,_,_,identity=initial_state(task,scenario,context_builder=build_context)
    try:
        prefix=len(env.records)
        while not env.done:env.step(corrective_action(env))
        if not env.success:raise ValueError('curriculum expert continuation failed: '+task.id+':'+str(scenario))
        return {'task_id':task.id,'scenario':scenario,'prefix_turns':prefix,
                'continuation_turns':len(env.records)-prefix,'total_turns':len(env.records),'passed':True}
    finally:env.close()


def frozen_inputs(cfg):
    tasks=training_tasks(cfg);dev=development_tasks(cfg)
    extra={}
    if cfg.get('curriculum_schema')=='fine_credit_ablation_v1':
        from .grpo_fine_study import method_record,probe_plan
        extra={'fine_credit_method.json':method_record(cfg),'common_update_probe_plan.json':probe_plan(cfg)}
    return {**extra,'curriculum_training_tasks.json':[asdict(t) for t in tasks],
            'curriculum_group_plan.json':[start_spec(cfg,i) for i in range(cfg['max_groups']+1)],
            'curriculum_isolation.json':assert_isolation(cfg),
            'curriculum_expert_proof.json':[expert_proof(t,s) for t in tasks+dev for s in (None,*SCENARIOS)],
            'targeted_evaluation_plan.json':targeted_plan(cfg),
            'targeted_recovery_contexts.json':[build_context(t,s,'development') for t in dev for s in TARGET_SCENARIOS]}



def summarize_targeted(episodes):
    def stats(rows):
        return {'episodes':len(rows),'successes':sum(e['success'] for e in rows),
                'actions':sum(len(e['turns']) for e in rows),
                'response_tokens':sum(len(r['generation']['response_token_ids']) for e in rows for r in e['turns'])}
    return {mode:{**stats([e for e in episodes if e['evaluation_mode']==mode]),
                  'by_scenario':{s:stats([e for e in episodes if e['evaluation_mode']==mode and e['scenario']==s])
                                 for s in TARGET_SCENARIOS}} for mode in ('greedy','sample')}
