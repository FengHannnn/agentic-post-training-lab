"""Record quality rejection without rerolling or admitting a failed checkpoint."""
import copy,json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,file_hash


def completed_evaluation(tmp_path,monkeypatch,alias=True):
    from test_grpo_formal_v2 import evaluation_fixture
    import apt_v2.grpo_curriculum as curriculum
    import grpo_vllm
    ev,tok,spec,data=evaluation_fixture(tmp_path,monkeypatch)
    cfg=ev.verify_inputs(tmp_path)
    tasks=ev.development_tasks(cfg)
    if alias:cfg['curriculum_schema']='alias_curriculum_v1'
    atomic_json(tmp_path/'config.json',cfg)
    data['config_sha256']=file_hash(tmp_path/'config.json')
    # Two different, valid native greedy responses to exactly the same prompt.
    generation=data['quality'][0]['generations'][1]
    generation['text']=json.dumps({'tool':'read_policy','arguments':{}})
    generation['response_token_ids']=[ord(c)+1 for c in generation['text']]+[0]
    monkeypatch.setattr(ev,'development_tasks',lambda cfg:tasks)
    monkeypatch.setattr(curriculum,'targeted_plan',lambda cfg:[])
    data.update(targeted_episodes=[],targeted_summary=curriculum.summarize_targeted([]))
    data['summary']=ev.summarize_evaluation(data['episodes'],cfg)
    atomic_json(tmp_path/'targeted_evaluation_plan.json',[])
    atomic_json(tmp_path/'evaluations/weighted_sft.json',data)
    atomic_json(tmp_path/'evaluations/weighted_sft-policy.json',spec)
    monkeypatch.setattr(grpo_vllm,'engine_and_tokenizer',lambda *a:pytest.fail('committed evaluation must not regenerate'))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    return ev,spec,tok,data


def test_committed_alias_quality_failure_is_preserved_and_recorded(tmp_path,monkeypatch):
    ev,spec,tok,data=completed_evaluation(tmp_path,monkeypatch)
    before=file_hash(tmp_path/'evaluations/weighted_sft.json')
    ev.evaluate(tmp_path,'weighted_sft',tmp_path/'evaluations/weighted_sft-policy.json')
    status=read(tmp_path/'evaluations/weighted_sft-quality-status.json')
    assert not status['passed'] and 'repeated_greedy_tokens_identical' in status['failed_checks']
    assert file_hash(tmp_path/'evaluations/weighted_sft.json')==before


def test_legacy_quality_failure_still_raises(tmp_path,monkeypatch):
    ev,spec,tok,data=completed_evaluation(tmp_path,monkeypatch,False)
    with pytest.raises(ValueError,match='quality'):
        ev.evaluate(tmp_path,'weighted_sft',tmp_path/'evaluations/weighted_sft-policy.json')


def test_structural_tampering_is_never_recorded_as_a_quality_outcome(tmp_path,monkeypatch):
    ev,spec,tok,data=completed_evaluation(tmp_path,monkeypatch)
    data['episodes'][0]['seed']+=1
    atomic_json(tmp_path/'evaluations/weighted_sft.json',data)
    with pytest.raises(ValueError):
        ev.evaluate(tmp_path,'weighted_sft',tmp_path/'evaluations/weighted_sft-policy.json')
    assert not (tmp_path/'evaluations/weighted_sft-quality-status.json').exists()


def test_recorded_missing_coalesce_really_fails_empty_refund_world():
    from apt_v2.grpo_alias_curriculum import development_tasks
    from apt_v2.tasks import metric_sql
    from apt_v2.independent_recovery_test import calendar_sql_check
    task=development_tasks()[3]
    sql=metric_sql(task,'prior_net_cents').replace('COALESCE(SUM(r.refund_cents),0)','SUM(r.refund_cents)')
    actual=calendar_sql_check(task,sql,['prior_net_cents'],allowed_split='development')
    assert not actual['passed'] and actual['worlds'][1]['actual']['prior_net_cents'] is None
    assert calendar_sql_check(task,metric_sql(task,'prior_net_cents'),['prior_net_cents'],allowed_split='development')['passed']


@pytest.mark.parametrize('failed_model',['weighted_sft','control','curriculum'])
@pytest.mark.parametrize('failure_kind',['standard','targeted'])
def test_any_original_quality_gate_failure_blocks_candidate_selection(failed_model,failure_kind):
    from audit_v2_grpo_alias import selection_with_quality
    from test_grpo_curriculum_v2 import counts
    summaries={'weighted_sft':counts(full=14),'control':counts(full=15),'curriculum':counts(full=14)}
    standard={key:{'passed':True} for key in summaries};targeted=copy.deepcopy(standard)
    assert selection_with_quality(summaries,standard,targeted)[1]['selected_candidate']=='control'
    (standard if failure_kind=='standard' else targeted)[failed_model]['passed']=False
    quality,selection=selection_with_quality(summaries,standard,targeted)
    assert not quality[failed_model] and selection['selected_candidate'] is None
    assert not selection['all_evaluation_quality_gates_passed']


def test_driver_finishes_evaluations_but_reports_quality_rejection(tmp_path,monkeypatch):
    from test_grpo_alias_driver_v2 import fixture
    driver,path,calls,output=fixture(tmp_path,monkeypatch,budget_stop=True)
    original=driver.run_logged
    def stage(command,**kwargs):
        code=original(command,**kwargs)
        if kwargs['log_path'].name.startswith('audit-attempt'):
            atomic_json(output/'alias_audit.json',{'passed':False,'integrity_passed':True})
        return code
    monkeypatch.setattr(driver,'run_logged',stage)
    driver.run(path)
    status=read(output/'run_status.json')
    assert status['error'] is None and status['stop_reason']=='completed_with_quality_rejections'
    assert status['evaluation_quality_passed'] is False
    assert len([c for c in calls if c.startswith('evaluate-')])==3


def test_targeted_semantic_failure_is_recorded_while_native_tampering_raises(tmp_path,monkeypatch):
    import evaluate_v2_grpo as ev
    from apt_v2.grpo_alias_curriculum import development_tasks,targeted_plan,summarize_targeted
    from test_grpo_curriculum_v2 import native_episode,NativeTokenizer
    cfg=read(ROOT/'configs/grpo-alias-control-v1.json');tasks={t.id:t for t in development_tasks()};tok=NativeTokenizer()
    plan=targeted_plan(cfg);rows=[]
    for item in plan:
        local=copy.deepcopy(cfg);local['sampling']['temperature']=item['temperature']
        episode=native_episode(tasks[item['task_id']],item['scenario'],tok,local,item['seed'],item['rollout_index'])
        episode.update(model_branch='grpo_g0048',evaluation_mode=item['mode']);rows.append(episode)
    atomic_json(tmp_path/'config.json',cfg);atomic_json(tmp_path/'targeted_evaluation_plan.json',plan)
    data={'targeted_episodes':rows,'targeted_summary':summarize_targeted(rows)}
    monkeypatch.setattr(ev,'calendar_episode_audit',lambda t,r,s,a,**k:{'environment_success':s,'calendar_consistent_success':False})
    check=ev.verify_targeted_evaluation(tmp_path,'grpo_g0048',{'policy':{'version':0}},tok,data=data)
    assert not check['passed'] and check['episodes_replayed']==32 and len(check['calendar'])==32
    rows[0]['seed']+=1
    with pytest.raises(ValueError):ev.verify_targeted_evaluation(tmp_path,'grpo_g0048',{'policy':{'version':0}},tok,data=data)
