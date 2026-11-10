"""One frozen retention warm start: broader train maintenance, lower LR, one epoch."""
import copy,json,math,random
from .grpo_repair_study import training_tasks,context,COMPOUND,W_SHA
from .grpo import initial_state

SCHEMA='repair_retention_v1'
PROFILE='retention_v1'
RUN_ID='grpo-sql-retention-study-v1-20261110'
RECIPE={'steps':15,'batch_size':4,'epochs':1,'learning_rate':2e-6,'seed':2026111053,
        'target_actions':12,'maintenance_actions':48}
STATES=('stale_evidence_repair','failed_check_repair','repeat_sql_error','checked_submit')


def validate_master(cfg):
    if (cfg.get('schema')!=SCHEMA or cfg.get('run_id')!=RUN_ID or cfg.get('sft')!=RECIPE
            or cfg.get('max_wall_seconds')!=16000 or cfg.get('total_authorized_seconds')!=21600
            or cfg.get('prior_elapsed_seconds')!=5578.0134817191865
            or cfg.get('packaging_reserve_seconds')!=180 or cfg.get('online_groups')!=36
            or cfg.get('k')!=4 or cfg.get('route')!='warm_start'):
        raise ValueError('frozen retention master contract differs')
    if cfg['max_wall_seconds']+cfg['prior_elapsed_seconds']>cfg['total_authorized_seconds']:
        raise ValueError('combined authorized budget exceeded')


def validate_prior(evidence,cfg):
    receipt=evidence['receipt'];status=evidence['final_status'];route=evidence['route']
    if (receipt['sha256']!='77d7e4cea2c9d0d481ee95a20583d00e8cc27a992c35ae46d983179bc00b9920'
            or receipt['elapsed_seconds_including_packaging']!=cfg['prior_elapsed_seconds']
            or status['stop_reason']!='b_gate_rejected' or status['error'] is not None
            or status['sft_updates']!=24 or status['actual_updates']!={'c':0}
            or route['route']!='warm_start' or not evidence['local_audit']['passed']
            or evidence['W_adapter_sha256']!=W_SHA):
        raise ValueError('audited prior route/budget evidence differs')
    return True


def supervision_dataset():
    from .dpo import corrective_action
    from grpo_repair_worker import verify_supervision
    examples=[]
    for task in training_tasks():
        for state in (COMPOUND,*STATES):
            scenario=None if state=='checked_submit' else state
            env,messages,_,_=initial_state(task,scenario,context_builder=context)
            try:
                if state=='checked_submit':
                    while True:
                        action=corrective_action(env)
                        if action['tool']=='submit':break
                        row=env.step(action)
                        if env.done:raise ValueError('submit maintenance prefix exhausted')
                        messages.extend([{'role':'assistant','content':json.dumps(action)},
                                         {'role':'user','content':json.dumps({'observation':row['observation']})}])
                action=corrective_action(env);row=env.step(action)
                if row['tool_error'] or not row['observation'].get('passed',True):
                    raise ValueError('productive retention action required')
                examples.append({'task_id':task.id,'split':'train','role':'target' if state==COMPOUND else 'maintenance',
                                 'state':state,'messages':copy.deepcopy(messages),'response':json.dumps(action),
                                 'observation':row['observation']})
            finally:env.close()
    verify_dataset(examples)
    verify_supervision(examples,expected_count=60)
    return examples


def verify_dataset(examples):
    tasks={t.id for t in training_tasks()}
    if len(examples)!=60:raise ValueError('60 distinct retention actions required')
    expected={(t,s) for t in tasks for s in (COMPOUND,*STATES)}
    if {(e['task_id'],e['state']) for e in examples}!=expected:raise ValueError('all 12 tasks and 5 states required')
    for e in examples:
        role='target' if e['state']==COMPOUND else 'maintenance'
        if e['split']!='train' or e['role']!=role:raise ValueError('train retention role differs')
        action=json.loads(e['response'])
        expected_tool={COMPOUND:'query_sql','stale_evidence_repair':'calculate_change',
                       'failed_check_repair':'check_result','repeat_sql_error':'query_sql','checked_submit':'submit'}[e['state']]
        if action['tool']!=expected_tool:raise ValueError('retention state corrective tool differs')
        if e['state']=='checked_submit' and not e['observation'].get('submitted'):
            raise ValueError('maintenance submission was not accepted')
    return True


def order(cfg):
    validate_master(cfg);rng=random.Random(RECIPE['seed']);items=list(range(60));rng.shuffle(items)
    return [items[i:i+4] for i in range(0,60,4)]
