import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,validate_config,initial_policy
from grpo_alias_common import matched_configs,training_decision,verify_training_stop
from apt_v2.grpo_alias_curriculum import (BRANCHES,TARGET_SCENARIOS,training_tasks,development_tasks,
    start_spec,context,frozen_inputs,exposure,choose_candidate)
from apt_v2.grpo_curriculum import build_context as routed_context,training_tasks as routed_tasks
from apt_v2.grpo import initial_state,replay
from apt_v2.grpo_credit import compute_credit,LOCAL
from test_grpo_curriculum_v2 import native_episode,NativeTokenizer


def config(arm='curriculum'):
    return read(ROOT/('configs/grpo-alias-control-v1.json' if arm=='control' else 'configs/grpo-alias-curriculum-arm-v1.json'))


def test_matched_local_only_configs_and_predeclared_dose():
    cfg,branches=matched_configs(ROOT/'configs/grpo-alias-curriculum-v1.json')
    plans={b:[start_spec(c,i) for i in range(48)] for b,c in branches.items()}
    assert cfg['max_wall_seconds']==21600
    for b,c in branches.items():
        validate_config(c);assert c['credit_method']==LOCAL and c['learning_rate']==1e-6
        assert len(routed_tasks(c))==12 and sum(t.id.startswith('train-') for t in training_tasks(c))==6
        assert exposure(c,48)=={'1':{'maintenance':12,'target':12},'2':{'maintenance':12,'target':12}}
        assert sum(p['scenario'] is None for p in plans[b])==12
        assert [p['task_id'] for p in plans[b][:24]]==[p['task_id'] for p in plans[b][24:]]
        assert {t.fault_mode for t in training_tasks(c)}=={'none','transient_query'}
        assert start_spec(c,48)['index']==48
    for i,(a,b) in enumerate(zip(plans['control'],plans['curriculum'])):
        assert {k:v for k,v in a.items() if k!='scenario'}=={k:v for k,v in b.items() if k!='scenario'}
        assert (a!=b)==(i<24 and i%2==1)
    for i in (-1,49,1.5):
        with pytest.raises(ValueError):start_spec(config(),i)


@pytest.mark.parametrize('arm',BRANCHES)
def test_all_contexts_experts_isolation_and_prefix_pairing(arm):
    c=config(arm);data=frozen_inputs(c)
    assert data['curriculum_isolation.json']['passed'] and len(data['curriculum_group_plan.json'])==49
    assert len(data['midpoint_plan.json'])==16 and len(data['targeted_evaluation_plan.json'])==32
    assert len(data['curriculum_expert_proof.json'])==160
    assert all(p['passed'] and p['total_turns']<=12 for p in data['curriculum_expert_proof.json'])
    for t in training_tasks(c)+development_tasks():
        simple=context(t,TARGET_SCENARIOS[0],t.split);compound=context(t,TARGET_SCENARIOS[1],t.split)
        assert len(simple['records'])==len(compound['records'])
        a=[r['row']['action'] for r in simple['records']];b=[r['row']['action'] for r in compound['records']]
        assert sum(x!=y for x,y in zip(a,b))==1
        assert simple['scenario']==TARGET_SCENARIOS[0] and compound['scenario']==TARGET_SCENARIOS[1]
        assert routed_context(t,TARGET_SCENARIOS[0],t.split)==simple
    assert development_tasks()[0].prior_month=='2031-12'


@pytest.mark.parametrize('scenario',TARGET_SCENARIOS)
@pytest.mark.parametrize('split',['train','development'])
def test_native_replay_local_credit_excludes_certified_prefix(scenario,split):
    c=config();t=(training_tasks(c) if split=='train' else development_tasks())[1];tok=NativeTokenizer()
    es=[native_episode(t,scenario,tok,c,123+i*1000,i,i<2) for i in range(4)]
    credit=compute_credit(es,method=LOCAL) if split=='train' else None
    if credit:assert credit['eligible']
    expected={'current_net_cents','change_cents'} if scenario.endswith('simplified') else {'current_net_cents','prior_net_cents','change_cents'}
    for i,e in enumerate(es):
        replay(t,e,tok,routed_context)
        rewarded={m for r in e['turns'] for m in r['new_verified_metrics']}
        assert rewarded==(expected if i<2 else set())
        if credit:
            assert sum(credit['local_progress_rewards'][i])==pytest.approx(.2*len(rewarded))
            assert len(credit['turn_advantages'][i])==len(e['turns'])
    bad=copy.deepcopy(es[0]);bad['start']['prefix'][0]['text']='tampered'
    with pytest.raises(ValueError):replay(t,bad,tok,routed_context)


@pytest.mark.parametrize('field,value',[('learning_rate',1e-5),('minimum_actual_updates',1),('max_groups',24),('curriculum_arm','adaptive'),('credit_method','outcome')])
def test_frozen_contract_rejects_unplanned_changes(field,value):
    c=config();c[field]=value
    with pytest.raises(ValueError):validate_config(c)


def test_budget_pending_midpoint_and_stop_arithmetic():
    assert training_decision(0,2200,[])['continue']
    for count in (22,24,46):
        d=training_decision(17000,2200,[],midpoint_pending=count<24)
        stop={'completed_groups':count,'maximum_groups':48,'reason':'budget_reserve',
              'policy':d['policy'],'budget_decision':d}
        assert verify_training_stop(stop,{'budget_policy':d['policy']},2200)==count
        bad=copy.deepcopy(stop);bad['budget_decision']['midpoint_pending']=not d['midpoint_pending']
        with pytest.raises(ValueError):verify_training_stop(bad,{'budget_policy':d['policy']},2200)
    assert training_decision(0,2200,[],True)['reserved_final_seconds']-training_decision(0,2200,[],False)['reserved_final_seconds']==600


def test_all_zero_signal_is_legal_only_with_unchanged_W_and_fresh_proof():
    from audit_v2_grpo import verified_zero_signal
    cfg=config();state=initial_policy(cfg);g={'policy':state,'credit':{'eligible':False}}
    tr={'updated':False,'before':state,'after':state,'metrics':{'skip_reason':'zero_variance'}}
    args=[cfg,state,[g,g],[tr],{'refresh_only_last_group':True},0]
    assert verified_zero_signal(*args)
    for index,value in [(0,{**cfg,'curriculum_schema':'fine_credit_ablation_v1'}),(2,[{**g,'credit':{'eligible':True}},g]),(3,[{**tr,'updated':True}]),(4,{'refresh_only_last_group':False}),(5,1)]:
        bad=copy.deepcopy(args);bad[index]=value;assert not verified_zero_signal(*bad)


def test_project_selection_stays_with_standard_success_and_greedy_guards():
    from test_grpo_curriculum_v2 import counts
    selected=choose_candidate({'weighted_sft':counts(full=16),'control':counts(full=15,recovery=70),'curriculum':counts(full=16,recovery=66)})
    assert selected['primary_metric']=='recovery' and selected['selected_candidate']=='curriculum'
    assert not selected['targeted_or_probe_used_for_selection']


@pytest.mark.parametrize('index',[3,23,27,47])
def test_training_worker_routes_new_contexts_and_sampling_capture(tmp_path,monkeypatch,index):
    import test_grpo_curriculum_v2 as fixture
    monkeypatch.setattr(fixture,'config',config)
    monkeypatch.setattr(fixture,'TARGET_SCENARIOS',TARGET_SCENARIOS)
    fixture.test_collection_worker_uses_fresh_tasks_and_native_custom_context(tmp_path,monkeypatch,index)
    group=read(tmp_path/'groups'/f'group-{index:04}.json')
    assert group['spec']==start_spec(config(),index) and group['credit']['method']==LOCAL


def test_same_forward_probability_probe_is_used_for_alias_schema(monkeypatch):
    import test_grpo_sampler_probe_v2 as fixture
    monkeypatch.setattr(fixture,'CFG',config())
    fixture.test_actual_same_forward_capture_overrides_cross_request_diagnostic_only(monkeypatch)


def test_phase_two_requires_both_successful_midpoints_across_attempts():
    from audit_v2_grpo_alias import verify_midpoint_order
    row=lambda name,code=0:{'stage':name,'exit_code':code}
    assert verify_midpoint_order([[row('midpoint-control'),row('midpoint-curriculum',1)],
                                  [row('midpoint-curriculum'),row('collect-control-0024')]])
    with pytest.raises(ValueError,match='before'):
        verify_midpoint_order([[row('midpoint-control'),row('collect-control-0024')]])


def test_root_audit_aggregates_compound_transfer_and_budget_stop(tmp_path,monkeypatch):
    import transformers
    import audit_v2_grpo_alias as audit
    from grpo_common import atomic_json
    from grpo_alias_common import training_decision
    from apt_v2.grpo_alias_curriculum import targeted_plan,summarize_targeted
    from test_grpo_curriculum_v2 import counts
    cfg,branches=matched_configs(ROOT/'configs/grpo-alias-curriculum-v1.json');atomic_json(tmp_path/'config.json',cfg)
    atomic_json(tmp_path/'run_manifest.json',{'source_sha256':{}})
    atomic_json(tmp_path/'execution_environment.json',{'gpu':'fixture-L40S'})
    decision=training_decision(17000,2200,[],True)
    atomic_json(tmp_path/'training_stop.json',{'completed_groups':2,'maximum_groups':48,'reason':'budget_reserve',
                'policy':cfg['budget_policy'],'budget_decision':decision})
    atomic_json(tmp_path/'midpoint_status.json',{'available_for_both':False,'group_cursor':24,'used_for_selection':False,
                'reason':'training_stopped_before_group_24'})
    atomic_json(tmp_path/'stage_journal.json',[])
    monkeypatch.setattr(audit,'frozen_inputs',lambda c:{})
    monkeypatch.setattr(audit,'audit_training',lambda *a,**k:{'actual_updates':0})
    monkeypatch.setattr(audit,'verify_objective_metrics',lambda *a:{'actual_updates_recomputed':0})
    monkeypatch.setattr(audit,'states_at',lambda child:[initial_policy(branches[child.name])]*3)
    monkeypatch.setattr(audit,'paired_evaluation',lambda *a:{'fixture':True})
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:object())
    summary=counts()
    for modes in summary.values():
        for stats in modes.values():stats.update(actions=0,response_tokens=0)
    monkeypatch.setattr(audit,'verify_evaluation',lambda *a,**k:{'passed':True,'summary':summary})
    def target_check(child,label,spec,tok):
        data=read(child/'evaluations'/(label+'.json'))
        return {'passed':True,'summary':summarize_targeted(data['targeted_episodes'])}
    monkeypatch.setattr(audit,'verify_targeted_evaluation',target_check)
    for b,c in branches.items():
        child=tmp_path/'branches'/b;atomic_json(child/'config.json',c);atomic_json(child/'execution_environment.json',{'gpu':'fixture-L40S'})
        for i in range(2):
            atomic_json(child/'groups'/f'group-{i:04}.json',{'spec':start_spec(c,i),'credit':{
                'mean':-.12,'population_std':0,'eligible':False,'turn_advantages':[[]]*4},'actions':4,'response_tokens':4})
            atomic_json(child/'transitions'/f'group-{i:04}.json',{'updated':False,'after':initial_policy(c),'metrics':{'skip_reason':'zero_variance'}})
        for label,cursor in ([('weighted_sft',0),('grpo_g0002',2)] if b=='control' else [('grpo_g0002',2)]):
            atomic_json(child/'evaluations'/(label+'-policy.json'),{'label':label,'group_cursor':cursor,'policy':initial_policy(c)})
            rows=[{**p,'evaluation_mode':p['mode'],'success':b=='curriculum' and p['scenario']==TARGET_SCENARIOS[1],
                    'turns':[]} for p in targeted_plan(c)]
            atomic_json(child/'evaluations'/(label+'.json'),{'elapsed_seconds':2200,'episodes':[],'targeted_episodes':rows})
    result=audit.audit(tmp_path,local=True)
    assert result['passed'] and result['completed_groups_per_branch']==2
    assert result['curriculum_transfer']['curriculum_minus_control_sample']==12
    assert not result['curriculum_transfer']['entered_compound_transfer_phase']
    assert result['targeted_paired']['curriculum_vs_control']['sample']['delta_successes']==12
    assert (tmp_path/'local_training_curve.csv').exists()
