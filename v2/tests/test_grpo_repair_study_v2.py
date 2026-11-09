import copy,json,sys
from dataclasses import asdict
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.tasks import make_tasks,metric_sql
from apt_v2.environment import Environment,execute_readonly
from apt_v2.sql_verifier import WORLDS,world,truth_in
from apt_v2.grpo_repair_study import *
from apt_v2.grpo import initial_state,replay,run_continuation,digest
from apt_v2.runtime import Generation
from grpo_common import read,validate_config


def config():return read(ROOT/'configs/grpo-sql-repair-branch-v1.json')


@pytest.mark.parametrize('metric',['current_net_cents','prior_net_cents'])
@pytest.mark.parametrize('index',[0,1,5,11])
def test_verifier_rejects_real_null_refund_and_accepts_equivalent_integer_sql(metric,index):
    old=[t for t in make_tasks() if t.split=='train'][index];task=RepairTask(**asdict(old))
    sql=metric_sql(task,metric);bad=sql.replace('COALESCE(SUM(r.refund_cents),0)','SUM(r.refund_cents)')
    for query,expected in ((bad,[]),(sql,[metric]),(sql.replace('COALESCE(SUM(r.refund_cents),0)','CAST(TOTAL(r.refund_cents) AS INTEGER)'),[metric])):
        env=Environment(task)
        try:
            env.query_attempts=1
            row=env.step({'tool':'query_sql','arguments':{'sql':query}})
            assert row['observation']['verified_metrics']==expected
        finally:env.close()
    assert old.public().keys()==task.public().keys() and 'verifier_version' not in task.request()


@pytest.mark.parametrize('name',WORLDS+('empty_gross',))
def test_worlds_preserve_oracle_integer_semantics_and_empty_sets(name):
    task=training_tasks()[0];db=world(task,name)
    try:
        gold=truth_in(db,task)
        for metric in ('current_net_cents','prior_net_cents'):
            rows,_=execute_readonly(db,metric_sql(task,metric));assert type(rows[0][metric]) is int and rows[0][metric]==gold[metric]
        for removed in ([task.current_month] if name=='no_current_refunds' else [task.prior_month] if name=='no_prior_refunds'
                        else [task.current_month,task.prior_month] if name=='no_target_refunds' else []):
            assert db.execute("SELECT COUNT(*) FROM refunds r JOIN orders o ON r.order_id=o.order_id WHERE o.status='paid' AND o.region=? AND o.segment=? AND r.refund_month=?",(task.region,task.segment,removed)).fetchone()[0]==0
    finally:db.close()


def test_version_preserves_legacy_task_schema_and_explicit_replay_prompt_binding():
    assert 'verifier_version' not in asdict(make_tasks()[0])
    task=training_tasks()[1];states_=[]
    for scenario in CONDITIONS:
        env,messages,sid,identity=initial_state(task,scenario,context_builder=context)
        try:
            assert env.verifier_version==VERSION and identity['messages']==messages and digest(identity)==sid
            states_.append((len(env.records),set(env.verified),messages,identity))
        finally:env.close()
    assert len({x[0] for x in states_})==1
    assert states_[0][1]==states_[1][1]==set() and states_[2][1]=={'prior_net_cents'}
    assert states_[1][2][-1]=={'role':'user','content':REMINDER}
    assert states_[0][3]['prefix']==states_[1][3]['prefix']


def oracle_episode(task,scenario,cfg,seed,ri,good=True):
    from apt_v2.dpo import corrective_action
    from apt_v2.grpo import parse_or_invalid
    def policy(messages):
        env=Environment(task)
        try:
            for m in messages[2:]:
                if m['role']=='assistant':env.step(parse_or_invalid(m['content']))
            action=corrective_action(env) if good else {'tool':'query_sql','arguments':{'sql':'SELECT 1'}}
            text=json.dumps(action)
            return Generation(text,'sample',[1],[2],None,[-.2],'length')
        finally:env.close()
    return run_continuation(task,scenario,policy,cfg['sampling'],0,seed,ri,.01,context)


def diagnostic_groups(raw=False,hint=False):
    cfg=config();master=read(ROOT/'configs/grpo-sql-repair-study-v1.json');tasks={t.id:t for t in diagnostic_tasks()};groups=[]
    for spec in diagnostic_plan(master):
        good=(raw if spec['scenario']==COMPOUND else hint if spec['scenario']==REMINDED else True)
        episodes=[oracle_episode(tasks[spec['task_id']],spec['scenario'],cfg,spec['seed']+ri*1000,ri,good) for ri in range(4)]
        groups.append({'spec':spec,'episodes':episodes,'credit':compute_credit(episodes,4,1e-8,LOCAL)})
    return groups


@pytest.mark.parametrize('raw,hint,route',[(True,True,'online_plain'),(False,True,'online_reminder'),(False,False,'warm_start')])
def test_route_from_actual_replayed_k4_groups(raw,hint,route):
    groups=diagnostic_groups(raw,hint);assert route_diagnostic(groups)['route']==route
    e=groups[8]['episodes'][0];task=next(t for t in diagnostic_tasks() if t.id==e['task_id']);replay(task,e,context_builder=context)
    bad=copy.deepcopy(e);bad['start']['messages'][-1]['content']='hidden injected text'
    with pytest.raises(ValueError):replay(task,bad,context_builder=context)
    bad=copy.deepcopy(groups);bad[0]['credit']['eligible']=not bad[0]['credit']['eligible']
    with pytest.raises(ValueError):route_diagnostic(bad)


def test_schedule_pairing_fading_isolation_and_supervision_exposure():
    from grpo_repair_worker import supervision_dataset,verify_supervision,sft_order
    cfg=config();validate_config(cfg);assert assert_isolation()['passed']
    control=[start_spec(dict(cfg,repair_arm='control'),i) for i in range(36)]
    curriculum=[start_spec(dict(cfg,repair_arm='curriculum'),i) for i in range(36)]
    for a,b in zip(control,curriculum):assert {k:v for k,v in a.items() if k!='scenario'}=={k:v for k,v in b.items() if k!='scenario'}
    assert [sum(r['scenario']==REMINDED for r in curriculum[i:i+12]) for i in (0,12,24)]==[6,3,0]
    assert sum(r['training_role']=='target' for r in control)==18
    data=supervision_dataset();assert verify_supervision(data) and len(data)==24
    bad=copy.deepcopy(data);bad[0]['response']='{"tool":"submit","arguments":{}}'
    with pytest.raises(ValueError):verify_supervision(bad)
    order=sft_order(read(ROOT/'configs/grpo-sql-repair-study-v1.json'));assert len(order)==24
    flat=sum(order,[]);assert all(flat.count(i)==4 for i in range(24))


def result(full=14,recovery=61,cg=4,rg=20,target=1,tg=0,first=1,quality=True):
    return {'effective_bf16_change':True,'spec':{'policy':{'updates':1,'adapter_sha256':str((full,recovery,cg,rg,target,tg,first,quality))}},
            'summary':{'sample':{'complete':{'successes':full},'recovery':{'successes':recovery}},
                       'greedy':{'complete':{'successes':cg},'recovery':{'successes':rg}}},
            'targeted_summary':{'sample':{'successes':target,'first_action_certifications':first},'greedy':{'successes':tg}},
            'evaluation_quality_passed':quality}


def test_gates_require_global_gain_compound_transfer_and_separate_grpo_increment():
    w=result();b=result(full=15,target=3,first=4)
    assert b_gate(w,b)['passed']
    assert not b_gate(w,result(first=2))['passed']
    assert candidate_checks(w,b)['passed'] and not candidate_checks(w,b,b)['passed']
    assert candidate_checks(w,result(full=16,target=4,first=5),b)['passed']
    assert not candidate_checks(w,result(full=15,target=2))['passed']
    assert not candidate_checks(w,result(full=15,target=3,quality=False))['passed']
    assert not candidate_checks(w,result(full=15,target=3,recovery=60))['passed']


def test_cross_month_wrong_join_is_rejected_and_maintenance_covers_both_pools():
    from apt_v2.environment import Environment
    from apt_v2.tasks import metric_sql
    for task in training_tasks():
        sql=metric_sql(task,'current_net_cents')
        bad=sql.replace("WHERE o.status='paid'", "WHERE o.order_month='"+task.current_month+"' AND o.status='paid'")
        env=Environment(task)
        try:
            env.query_attempts=1
            row=env.step({'tool':'query_sql','arguments':{'sql':bad}})
            assert row['observation']['verified_metrics']==[]
        finally:env.close()
    c=config();plan=[start_spec(c,i) for i in range(36)];maintenance=[p for p in plan if p['training_role']=='maintenance']
    assert sum(p['scenario'] is None for p in maintenance)==9
    assert {p['scenario'] for p in maintenance if p['scenario']}==set(STANDARD)
    for complete in (True,False):
        assert {p['task_id'].startswith('train-') for p in maintenance if (p['scenario'] is None)==complete}=={True,False}
