"""Test-only recovery contexts, native continuations and calendar SQL challenges."""
import json
from dataclasses import asdict
from .tasks import make_tasks,connect,metric_sql
from .environment import Environment,METRICS,execute_readonly
from .runtime import SYSTEM
from .dpo import action_from_text,corrective_action,TEMPORARY_ERROR
from .dpo_recovery import SCENARIOS,build_context
from .dpo_recovery_training import validate_generation,validate_training_pairs
from .recovery_diagnostics import diagnose_action,messages_id

NOVEL_SCENARIOS=('both_sql_unverified','both_sql_unverified_repeated_evidence')
TEST_SCENARIOS=(*SCENARIOS,*NOVEL_SCENARIOS)

def independent_tasks(pairs,examples):
    validate_training_pairs(pairs)
    tasks=[t for t in make_tasks() if t.split=='test']
    if len(tasks)!=4:raise ValueError('four fixed test tasks required')
    train=[t for t in make_tasks() if t.split=='train'];other=[t for t in make_tasks() if t.split!='test']
    if any(e['split']!='train' or (e['task_id'],e['family_id']) not in {(t.id,t.family_id) for t in train} for e in examples):
        raise ValueError('train-only example provenance required')
    for key in ('id','family_id','seed'):
        if {getattr(t,key) for t in tasks}&{getattr(t,key) for t in other}:raise ValueError('test task family or seed overlaps')
    if {t.id for t in tasks}&{p['task_id'] for p in pairs}:raise ValueError('test pair leakage')
    return tasks

def build_test_context(task,scenario):
    if task.split!='test' or scenario not in TEST_SCENARIOS:raise ValueError('explicit test-only scenario required')
    if scenario in SCENARIOS:return build_context(task,scenario,allowed_split='test')
    env=Environment(task);records=[];messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]
    def step(action):
        text=json.dumps(action)
        row=env.step(action);records.append({'text':text,'row':row})
        messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if row['tool_error']==TEMPORARY_ERROR:return step(action)
        return row
    try:
        step({'tool':'read_policy','arguments':{}});step({'tool':'inspect_schema','arguments':{}})
        ids={}
        for metric in ('current_net_cents','prior_net_cents'):
            row=step({'tool':'query_sql','arguments':{'sql':'SELECT SUM(gross_cents) AS '+metric+' FROM orders'}})
            if row['observation'].get('verified_metrics'):raise ValueError('unverified monthly SQL injection required')
            ids[metric]=row['observation']['evidence_id']
        rejected={'tool':'calculate_change','arguments':{'current_evidence':ids['current_net_cents'],'prior_evidence':ids['prior_net_cents']}}
        if scenario=='both_sql_unverified_repeated_evidence':step(rejected);step(rejected)
        if env.verified or env.done:raise ValueError('both monthly metrics must remain unverified')
        chosen=corrective_action(env);row=env.step(chosen)
        if row['tool_error'] is not None or not row['new_verified_metrics']:raise ValueError('expert first recovery query required')
        return {'context_id':task.id+':'+scenario,'task_id':task.id,'family_id':task.family_id,'split':'test','scenario':scenario,
            'prefix_source':'scripted_error_injection','records':records,'messages':messages,'chosen':json.dumps(chosen),
            'chosen_observation':row['observation'],'scripted_rejected':json.dumps(rejected)}
    finally:env.close()

def test_contexts(tasks):return [build_test_context(t,s) for t in tasks for s in TEST_SCENARIOS]

def diagnose_test_action(task,context,text):
    return diagnose_action(task,context,text,context_builder=build_test_context,allowed_splits=('test',))

def context_environment(task,context):
    if context!=build_test_context(task,context['scenario']):raise ValueError('test recovery context differs')
    env=Environment(task)
    try:
        for item in context['records']:
            if env.step(action_from_text(item['text']))!=item['row']:raise ValueError('test prefix replay differs')
        return env
    except Exception:env.close();raise

def run_test_recovery(task,context,policy):
    if context['scenario'] not in NOVEL_SCENARIOS:raise ValueError('novel test continuation required')
    env=context_environment(task,context);messages=[dict(m) for m in context['messages']];rows=[];prefix_errors=env.tool_errors
    try:
        while not env.done:
            sample=policy(messages);row=env.step(action_from_text(sample.text));row['generation']=asdict(sample);rows.append(row)
            messages.extend([{'role':'assistant','content':sample.text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        return {'task_id':task.id,'family_id':task.family_id,'split':'test','context_id':context['context_id'],'scenario':context['scenario'],
            'success':env.success,'turns':rows,'tool_errors':env.tool_errors-prefix_errors,'total_tool_errors':env.tool_errors,
            'termination':rows[-1]['termination'],'artifact':rows[-1]['observation'].get('artifact') if env.success else None}
    except Exception as exc:
        exc.episode_evidence={'task_id':task.id,'context_id':context['context_id'],'status':'interrupted','turns':rows,'error':str(exc)}
        raise
    finally:env.close()

def replay_test_recovery(task,context,episode,tokenizer,native_mode):
    if episode['task_id']!=task.id or episode['family_id']!=task.family_id or episode['split']!='test' or episode['context_id']!=context['context_id'] or episode['scenario']!=context['scenario']:
        raise ValueError('test recovery episode identity differs')
    env=context_environment(task,context);messages=[dict(m) for m in context['messages']];prefix_errors=env.tool_errors
    try:
        if not episode['turns']:raise ValueError('nonempty test continuation required')
        for row in episode['turns']:
            g=row['generation'];validate_generation(tokenizer,messages,g,native_mode)
            actual=env.step(action_from_text(g['text']))
            if actual!={k:v for k,v in row.items() if k not in ('generation','sampling_seed')}:raise ValueError('test continuation replay differs')
            messages.extend([{'role':'assistant','content':g['text']},{'role':'user','content':json.dumps({'observation':row['observation']})}])
        if not env.done or env.success!=episode['success'] or env.tool_errors-prefix_errors!=episode['tool_errors'] or env.tool_errors!=episode['total_tool_errors'] or env.records[-1]['termination']!=episode['termination']:
            raise ValueError('test continuation completion differs')
        artifact=env.records[-1]['observation'].get('artifact') if env.success else None
        if artifact!=episode['artifact']:raise ValueError('test continuation artifact differs')
        return len(episode['turns'])
    finally:env.close()

def calendar_sql_check(task,sql,metrics,*,allowed_split='test'):
    if allowed_split not in ('test','development') or task.split!=allowed_split or not metrics or any(m not in METRICS for m in metrics):raise ValueError('test SQL metrics required')
    worlds=[]
    for world in (1,2):
        db=connect(task,world)
        try:
            refs={}
            for label,month in [('current',task.current_month),('prior',task.prior_month)]:
                refs[label]=db.execute("SELECT r.refund_id,o.gross_cents FROM refunds r JOIN orders o ON r.order_id=o.order_id WHERE o.status='paid' AND o.region=? AND o.segment=? AND o.order_month=? ORDER BY r.refund_id LIMIT 1",
                    (task.region,task.segment,month)).fetchone()
                if refs[label] is None:raise ValueError('refunded current and prior source orders required')
            if world==1:
                for label,month in [('prior',task.current_month),('current',task.prior_month)]:
                    rid,gross=refs[label];db.execute('UPDATE refunds SET refund_month=?,refund_cents=? WHERE refund_id=?',(month,max(1,min(1237,gross//4)),rid))
            else:
                db.execute("UPDATE refunds SET refund_month=? WHERE order_id IN (SELECT order_id FROM orders WHERE status='paid' AND region=? AND segment=?)",
                    (task.current_month,task.region,task.segment))
            gold={m:execute_readonly(db,metric_sql(task,m))[0][0][m] for m in ('current_net_cents','prior_net_cents')}
            gold['change_cents']=gold['current_net_cents']-gold['prior_net_cents']
            try:
                rows,_=execute_readonly(db,sql);error=None
                actual={m:rows[0].get(m) if len(rows)==1 else None for m in metrics}
            except ValueError as exc:actual={m:None for m in metrics};error=str(exc)
            passed={m:error is None and type(actual[m]) is int and actual[m]==gold[m] for m in metrics}
            worlds.append({'world':world,'expected':{m:gold[m] for m in metrics},'actual':actual,'metric_passed':passed,'error':error})
        finally:db.close()
    per_metric={m:all(w['metric_passed'][m] for w in worlds) for m in metrics}
    return {'metrics':list(metrics),'passed':all(per_metric.values()),'metric_passed':per_metric,'worlds':worlds}

def calendar_episode_audit(task,records,success,artifact,*,allowed_split='test'):
    queries=[];registry={}
    for row in records:
        eid=row['observation'].get('evidence_id')
        if eid:registry[eid]=row
        if row['action'].get('tool')=='query_sql' and row['observation'].get('verified_metrics'):
            metrics=row['observation']['verified_metrics']
            queries.append({'turn':row['turn'],'evidence_id':eid,'check':calendar_sql_check(task,row['action']['arguments']['sql'],metrics,allowed_split=allowed_split)})
    checks={q['evidence_id']:q['check'] for q in queries};needed={};visiting=set()
    def need(eid,metric):
        if (eid,metric) in visiting:return
        visiting.add((eid,metric));row=registry.get(eid)
        if row is None:needed.setdefault('<missing>',set()).add(metric);return
        if row['action'].get('tool')=='calculate_change' and metric=='change_cents':
            args=row['action']['arguments'];need(args.get('current_evidence'),'current_net_cents');need(args.get('prior_evidence'),'prior_net_cents')
        else:needed.setdefault(eid,set()).add(metric)
    if success:
        for m in METRICS:need(artifact['evidence'][m],m)
    final_sources=[{'evidence_id':eid,'metrics':sorted(ms),'passed':all(checks.get(eid,{}).get('metric_passed',{}).get(m,False) for m in ms)} for eid,ms in sorted(needed.items())]
    return {'environment_success':success,'verified_query_count':len(queries),'calendar_queries_passed':sum(q['check']['passed'] for q in queries),
        'query_checks':queries,'final_sources':final_sources,
        'calendar_consistent_success':bool(success and final_sources and all(r['passed'] for r in final_sources))}
