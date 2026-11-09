import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.grpo_curriculum import (SCHEMA,SCENARIOS,TARGET_SCENARIOS,training_tasks,development_tasks,
    assert_isolation,start_spec,build_context,expert_proof,targeted_plan,frozen_inputs,summarize_targeted)
from apt_v2.grpo import initial_state,run_continuation,replay,digest,group_advantages
from apt_v2.grpo_evaluation import evaluation_plan,summarize_evaluation,paired_evaluation
from apt_v2.dpo_recovery import build_context as legacy_context,SCENARIOS as LEGACY_SCENARIOS
from apt_v2.dpo import corrective_action
from apt_v2.rewards import training_samples
from grpo_common import validate_config,atomic_json,file_hash
from grpo_curriculum_common import resume_spent,select_candidate,matched_configs
from test_grpo_confirmation_v2 import NativeTokenizer


def config(groups=48):
    cfg=json.loads((ROOT/'configs/grpo-c3-outcome-v1.json').read_text())
    cfg.update(run_id='test-outcome-curriculum-control',study_phase='outcome_curriculum',curriculum_schema=SCHEMA,
               max_groups=groups,max_updates=groups,group_evaluation_cursors=[groups],
               max_wall_seconds=14400 if groups==24 else 21600,formal_gpu_budget_seconds=14400 if groups==24 else 21600,
               recovery_scenarios=list(SCENARIOS),seed=2026110717,
               closing_evaluation={'enabled':False},targeted_evaluation={'enabled':True,'seed':2026110719,'samples_per_context':1})
    return cfg


def native_episode(task,scenario,tok,cfg,seed=123,index=0,success=True):
    env,_,_,_=initial_state(task,scenario,context_builder=build_context)
    def policy(messages):
        action=corrective_action(env) if success else {'tool':'calculate_change','arguments':{'current_evidence':'invalid','prior_evidence':'invalid'}}
        env.step(action);return tok.generation(messages,json.dumps(action),cfg['sampling'])
    try:return run_continuation(task,scenario,policy,cfg['sampling'],0,seed,index,context_builder=build_context)
    finally:env.close()


@pytest.mark.parametrize('groups',[24,48])
def test_isolated_balanced_schedule_and_expert_budget(groups):
    cfg=config(groups);validate_config(cfg);report=assert_isolation(cfg)
    assert report['train_tasks']==groups//2 and report['development_tasks']==4 and report['c4_tasks_excluded']==12
    tasks=training_tasks(cfg)
    assert sum(t.id.startswith('train-') for t in tasks)==groups//4
    plan=[start_spec(cfg,i) for i in range(groups)]
    assert sum(p['kind']=='complete' for p in plan)==groups//2
    assert {p['scenario'] for p in plan if p['scenario']}==set(SCENARIOS)
    assert all(plan[i]['task_id']==plan[i+1]['task_id'] for i in range(0,groups,2))
    proofs=[expert_proof(t,s) for t in tasks+development_tasks() for s in (None,*SCENARIOS)]
    assert len(proofs)==(groups//2+4)*11 and all(p['passed'] for p in proofs)
    assert max(p['total_turns'] for p in proofs)==11
    assert development_tasks()[0].prior_month=='2027-12'


@pytest.mark.parametrize('scenario',TARGET_SCENARIOS)
def test_targeted_native_replay_and_prefix_excluded_from_loss(scenario):
    cfg=config();task=training_tasks(cfg)[3];tok=NativeTokenizer()
    episodes=[native_episode(task,scenario,tok,cfg,123+i*1000,i,i<2) for i in range(4)]
    assert all(replay(task,e,tok,build_context)==len(e['turns']) for e in episodes)
    credit=group_advantages(episodes)
    assert credit['eligible'] and credit['advantages'][0]>0 and credit['advantages'][2]<0
    for e in episodes:
        samples=training_samples(e)
        assert len(samples)==len(e['turns']) and samples[0]['turn']==e['prefix_turns']
        assert e['return']==pytest.approx(float(e['success'])-.01*e['total_turns'])
    bad=copy.deepcopy(episodes[0]);bad['start']['prefix'][0]['text']='bad'
    with pytest.raises(ValueError):replay(task,bad,tok,build_context)
    bad=copy.deepcopy(episodes[0]);bad['turns'][0]['generation']['prompt_token_ids']=[12345]
    with pytest.raises(ValueError):replay(task,bad,tok,build_context)


def test_real_error_types_and_unverified_evidence():
    task=training_tasks(config())[0]
    alias=build_context(task,'missing_alias_pair')['records']
    assert all(not r['row']['observation'].get('verified_metrics') for r in alias[2:4])
    assert alias[-1]['row']['tool_error']
    schema=build_context(task,'schema_query_repeat')['records']
    assert all(r['row']['tool_error'] for r in schema[2:4])
    stale=build_context(task,'requery_stale_change')['records']
    assert stale[-1]['row']['observation']['passed'] is False
    assert 'current_net_cents' in stale[-2]['row']['observation']['verified_metrics']
    assert stale[-1]['row']['action']['arguments']['evidence']['change_cents']==stale[-2]['row']['observation']['evidence_id']


def test_original_recovery_contexts_and_default_state_unchanged():
    task=training_tasks(config())[0]
    for scenario in LEGACY_SCENARIOS:
        assert build_context(task,scenario)==legacy_context(task,scenario)
        a=initial_state(task,scenario);b=initial_state(task,scenario,context_builder=build_context)
        try:assert a[1:]==b[1:]
        finally:a[0].close();b[0].close()


def test_standard_132_and_targeted_24_are_separate_and_fresh():
    cfg=config();standard=evaluation_plan(cfg);targeted=targeted_plan(cfg)
    assert len(standard)==132 and len(targeted)==24
    assert {p['task_id'] for p in standard}=={t.id for t in development_tasks()}
    assert not {p['scenario'] for p in targeted}&{p['scenario'] for p in standard}
    assert len({(p['task_id'],p['scenario'],p['mode'],p['rollout_index']) for p in targeted})==24
    old=json.loads((ROOT/'configs/grpo-c3-outcome-v1.json').read_text())
    assert len(evaluation_plan(old))==132 and targeted_plan(old)==[]


def test_new_development_summaries_and_pairs_keep_task_ids():
    cfg=config();tok=NativeTokenizer();rows=[]
    for p in evaluation_plan(cfg):
        # Small synthetic summaries have the frozen keys; native replay is tested separately.
        rows.append({**p,'evaluation_mode':p['mode'],'success':True,'turns':[],'tool_errors':0})
    summary=summarize_evaluation(rows,cfg)
    assert summary['sample']['complete']['successes']==16 and summary['sample']['recovery']['successes']==84
    assert set(summary['sample']['complete']['by_task'])=={t.id for t in development_tasks()}
    paired=paired_evaluation(rows,rows,cfg)
    assert paired['sample']['complete']['delta_successes']==0 and len(paired['sample']['complete']['by_task'])==4


@pytest.mark.parametrize('field,value',[('learning_rate',3e-6),('max_wall_seconds',30000),('curriculum_schema','unknown'),('credit_method','potential_rtg_depth_v1')])
def test_invalid_curriculum_contract_rejected(field,value):
    cfg=config();cfg[field]=value
    with pytest.raises(ValueError):validate_config(cfg)


def counts(full=14,recovery=65,gfull=4,grecovery=20):
    return {'sample':{'complete':{'successes':full,'episodes':16},'recovery':{'successes':recovery,'episodes':84}},
            'greedy':{'complete':{'successes':gfull,'episodes':4},'recovery':{'successes':grecovery,'episodes':28}}}


def test_selection_uses_primary_gain_and_greedy_guards_only():
    r=select_candidate({'weighted_sft':counts(),'control':counts(full=15),'lr10x':counts(full=16,grecovery=19)})
    assert r['selected_candidate']=='control' and r['primary_metric']=='complete'
    r=select_candidate({'weighted_sft':counts(full=16),'control':counts(full=16,recovery=66),
                        'lr10x':counts(full=16,recovery=67)})
    assert r['selected_candidate']=='lr10x' and r['primary_metric']=='recovery'
    r=select_candidate({'weighted_sft':counts(),'control':counts(),'lr10x':counts(full=13,recovery=84)})
    assert r['selected_candidate'] is None


def test_cumulative_budget_includes_unrecorded_active_stage():
    assert resume_spent({'elapsed_seconds':100,'active_started_unix':900},{'elapsed_seconds':90},now=1000)==200
    assert resume_spent({'elapsed_seconds':100,'active_started_unix':None},{'elapsed_seconds':120},now=1000)==120


def driver_fixture(tmp_path,monkeypatch,fail_once):
    from types import SimpleNamespace
    import run_v2_grpo_curriculum as driver
    root=tmp_path/'v2';cfg={'run_id':'test-outcome-curriculum','max_wall_seconds':21600,'packaging_reserve_seconds':180,'groups_per_branch':2}
    branches={b:{'run_id':'test-'+b,'adapter_sha256':'W','max_groups':2} for b in ['control','lr10x']}
    path=root/'configs/study.json';atomic_json(path,cfg)
    atomic_json(root/'grpo_curriculum_release_manifest.json',{'files':{'configs/study.json':file_hash(path)}})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'matched_configs',lambda p:(cfg,branches))
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/study.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='NVIDIA L40S, NEW-UUID, driver',stderr=''))
    calls=[];failed=[False]
    def stage(command,**kwargs):
        name=kwargs['log_path'].name.split('-attempt')[0];calls.append(name)
        kwargs['log_path'].write_text('Progress 100%\r',encoding='utf8');target=Path(command[-1])
        if name=='collect-lr10x-0000' and fail_once and not failed[0]:failed[0]=True;return 1
        if name.startswith('prepare-'):
            atomic_json(target/'input_binding.json',{});atomic_json(target/'execution_environment.json',{'gpu_uuid':'NEW-UUID'})
        if name.startswith('collect-') or name.startswith('refresh-'):
            index=int(command[command.index('--group-index')+1]);atomic_json(target/'groups'/f'group-{index:04}.json',{'index':index})
        if name.startswith('update-'):
            index=int(command[command.index('--group-index')+1]);before=json.loads(Path(command[command.index('--policy-file')+1]).read_text())
            atomic_json(target/'transitions'/f'group-{index:04}.json',{'index':index,'before':before,'after':before,'updated':False,
                'config_sha256':file_hash(target/'config.json'),'group_sha256':file_hash(target/'groups'/f'group-{index:04}.json')})
        if name.startswith('evaluate-'):
            label=command[command.index('--label')+1];atomic_json(target/'evaluations'/(label+'.json'),{})
        if name=='audit':atomic_json(target/'curriculum_audit.json',{'passed':True})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    return driver,path,calls,root/'artifacts'/cfg['run_id']


def test_driver_alternates_independent_branches_and_evaluates_three_models(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,False)
    archive=driver.run(path)
    assert archive.exists()
    assert [n for n in calls if n.startswith('update-')]==['update-control-0000','update-lr10x-0000','update-control-0001','update-lr10x-0001']
    assert len([n for n in calls if n.startswith('evaluate-')])==3
    status=json.loads((output/'run_status.json').read_text())
    assert status['error'] is None and status['committed_groups']=={'control':2,'lr10x':2}
    assert json.loads((output/'runner_state.json').read_text())['active_started_unix'] is None


def test_driver_resume_keeps_other_branch_commits_and_budget(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError):driver.run(path)
    prior=json.loads((output/'run_status.json').read_text())['elapsed_seconds']
    archive=driver.run(path,True)
    assert archive.name.endswith('attempt-002.zip') and calls.count('update-control-0000')==1
    status=json.loads((output/'run_status.json').read_text())
    assert status['error'] is None and status['elapsed_seconds']>=prior and status['max_wall_seconds']==21600



def test_budget_profiles_are_matched_except_learning_rate(tmp_path,monkeypatch):
    import grpo_curriculum_common as common
    original_root=common.ROOT
    for hours in (4,6,7):
        cfg,branches=common.configuration_profile(hours)
        assert cfg['groups_per_branch']==(24 if hours==4 else 48)
        root=tmp_path/str(hours);path=root/'configs/grpo-outcome-curriculum-v1.json'
        atomic_json(path,cfg)
        for b,branch in branches.items():atomic_json(root/cfg['branches'][b],branch)
        monkeypatch.setattr(common,'ROOT',root)
        assert common.matched_configs(path)==(cfg,branches)
        changed=copy.deepcopy(branches['lr10x']);changed['beta']=.03
        atomic_json(root/cfg['branches']['lr10x'],changed)
        with pytest.raises(ValueError,match='only in learning rate'):common.matched_configs(path)
        monkeypatch.setattr(common,'ROOT',original_root)


def test_saturated_full_success_is_preserved_when_recovery_selects():
    result=select_candidate({'weighted_sft':counts(full=16),'control':counts(full=15,recovery=68),
                             'lr10x':counts(full=16,recovery=66)})
    assert result['selected_candidate']=='lr10x' and not result['sample_nonregression_guards']['control']


def test_targeted_evaluator_replays_all_24_native_episodes_and_rejects_seed_tamper(tmp_path):
    from evaluate_v2_grpo import verify_targeted_evaluation
    from apt_v2.grpo_evaluation import plan_key
    cfg=config();tok=NativeTokenizer();tasks={t.id:t for t in development_tasks()};rows=[]
    for p in targeted_plan(cfg):
        local_cfg=copy.deepcopy(cfg);local_cfg['sampling']['temperature']=p['temperature']
        e=native_episode(tasks[p['task_id']],p['scenario'],tok,local_cfg,p['seed'],p['rollout_index'])
        e.update(model_branch='weighted_sft',evaluation_mode=p['mode']);rows.append(e)
    atomic_json(tmp_path/'config.json',cfg);atomic_json(tmp_path/'targeted_evaluation_plan.json',targeted_plan(cfg))
    data={'targeted_episodes':rows,'targeted_summary':summarize_targeted(rows)}
    result=verify_targeted_evaluation(tmp_path,'weighted_sft',{'policy':{'version':0}},tok,data)
    assert result['passed'] and result['episodes_replayed']==24
    bad=copy.deepcopy(data);bad['targeted_episodes'][0]['seed']+=1
    with pytest.raises(ValueError,match='sampling/model'):verify_targeted_evaluation(tmp_path,'weighted_sft',{'policy':{'version':0}},tok,bad)
    bad=copy.deepcopy(data);bad['targeted_episodes'].pop()
    with pytest.raises(ValueError,match='coverage'):verify_targeted_evaluation(tmp_path,'weighted_sft',{'policy':{'version':0}},tok,bad)


def test_full_profile_new_errors_cover_old_and_fresh_families():
    cfg=config()
    rows=[start_spec(cfg,i) for i in range(cfg['max_groups'])]
    for s in TARGET_SCENARIOS:
        starts=[p for p in rows if p['scenario']==s]
        assert any(p['task_id'].startswith('train-') for p in starts)
        assert any(p['task_id'].startswith('curriculum-train-') for p in starts)


@pytest.mark.parametrize('index',[23,27,39])
def test_collection_worker_uses_fresh_tasks_and_native_custom_context(tmp_path,monkeypatch,index):
    import grpo_worker as worker
    import grpo_vllm
    from dataclasses import asdict
    cfg=config();spec=start_spec(cfg,index);task=worker.task_for(spec['task_id'],cfg)
    assert task.id.startswith('curriculum-train-') and spec['scenario'] in TARGET_SCENARIOS
    tok=NativeTokenizer();adapter=tmp_path/'initial/adapter';adapter.mkdir(parents=True)
    (adapter/'adapter_model.safetensors').write_bytes(b'mocked-GPU-policy')
    before={'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':file_hash(adapter/'adapter_model.safetensors')}
    atomic_json(tmp_path/'config.json',cfg)
    monkeypatch.setattr(worker,'verify_inputs',lambda *a,**k:cfg)
    tensors={str(i):{'fixture':i} for i in range(112)}
    class Engine:
        def collective_rpc(self,*a,**k):
            return [{'passed':True,'adapter_sha256':before['adapter_sha256'],'registered_ids':[1],
                     'active_slots':[1],'tensor_count':112,'tensors':tensors,'loaded_tensor_sha256':digest(tensors)}]
    engine=Engine();environments=[]
    class Policy:
        def __init__(self,engine,tok,lora,sampling,seed,action):
            self.env,_,_,_=initial_state(task,spec['scenario'],context_builder=build_context)
            environments.append(self.env);self.calls=0;self.first=None;self.sampling=sampling;self.action=action
        def __call__(self,messages):
            action=corrective_action(self.env);self.env.step(action);self.calls+=1
            generation=tok.generation(messages,json.dumps(action),self.sampling)
            if self.first is None:self.first=asdict(generation)
            self.action(len(generation.response_token_ids));return generation
    def probe(engine,tok,lora,first,cfg,seed=None,worker_proof=None):
        return {'tokens':first['response_token_ids'],'behavior_q':first['behavior_token_log_probs'],
                'teacher_forced_vllm_logp':first['behavior_token_log_probs'],'gate':cfg['probability_probe_max_error'],
                'passed':True,'temperature':1,'top_p':1,'top_k':-1,'max_absolute_error':0.,'extra_generated_tokens':0}
    monkeypatch.setattr(grpo_vllm,'engine_and_tokenizer',lambda *a:(engine,tok,None))
    monkeypatch.setattr(grpo_vllm,'BehaviorPolicy',Policy);monkeypatch.setattr(grpo_vllm,'probability_probe',probe)
    monkeypatch.setattr(grpo_vllm,'shutdown',lambda engine:[e.close() for e in environments])
    worker.collect(tmp_path,index,before)
    group=worker.verify_group(tmp_path,index,before,tok)
    assert group['spec']==spec and len(group['episodes'])==4
    assert all(e['success'] and e['task_id']==task.id for e in group['episodes'])
    path=tmp_path/'groups'/f'group-{index:04}.json'
    bad=json.loads(path.read_text());bad['episodes'][0]['start']['prefix'][0]['text']='tampered';atomic_json(path,bad)
    with pytest.raises(ValueError):worker.verify_group(tmp_path,index,before,tok)
