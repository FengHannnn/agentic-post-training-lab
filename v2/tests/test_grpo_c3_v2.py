import copy
import json
import sys
from pathlib import Path
from dataclasses import asdict
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.grpo_credit import compute_credit,action_advantage,FINE
from apt_v2.grpo import run_continuation,replay,group_advantages
from apt_v2.period_closing import (closing_tasks,ClosingEnvironment,expert_dependency_proof,next_sql,expected_next,TABLES,run_closing,replay_closing)
from apt_v2.environment import execute_readonly,ToolError
from apt_v2.tasks import metric_sql
from apt_v2.runtime import Generation
from test_grpo_v2 import episode,group,TASK,SAMPLING,tiny_model,scored_group


def test_outcome_credit_is_byte_for_byte_legacy():
    es=group();assert compute_credit(es)==group_advantages(es)


def test_fine_first_action_matches_outcome_and_depth_centers():
    es=group();outcome=group_advantages(es);credit=compute_credit(es,method=FINE)
    assert [a[0] for a in credit['turn_advantages']]==pytest.approx(outcome['advantages'])
    for d in credit['depth_statistics']:
        vals=[credit['turn_advantages'][i][d['relative_turn']] for i in d['active_episodes']]
        assert sum(vals)==pytest.approx(0.,abs=1e-7)
    assert any(len(set(round(v,6) for v in a))>1 for a in credit['turn_advantages'])


def combined_episode(index):
    cur,prior=metric_sql(TASK,'current_net_cents'),metric_sql(TASK,'prior_net_cents')
    actions=[{'tool':'read_policy','arguments':{}},{'tool':'inspect_schema','arguments':{}},
             {'tool':'query_sql','arguments':{'sql':f'SELECT a.current_net_cents,b.prior_net_cents FROM ({cur}) a CROSS JOIN ({prior}) b'}},
             {'tool':'read_policy','arguments':{}},{'tool':'calculate_change','arguments':{'current_evidence':'q1','prior_evidence':'q1'}}]
    from apt_v2.tasks import truth
    final={'values':truth(TASK),'evidence':{'current_net_cents':'q1','prior_net_cents':'q1','change_cents':'c2'},'policy_id':'revenue-v1'}
    actions += [{'tool':'check_result','arguments':final},{'tool':'submit','arguments':final}]
    count=[0]
    def policy(messages):
        a=actions[count[0]];count[0]+=1;return Generation(json.dumps(a))
    e=run_continuation(TASK,None,policy,SAMPLING,0,100+index*1000,index)
    for row in e['turns']:row['generation'].update(mode='sample',prompt_token_ids=[1,2,3],response_token_ids=[4,2],behavior_token_log_probs=[-1.,-1.],raw_token_log_probs=None,finish_reason='stop')
    assert e['success'] and replay(TASK,e)==7
    return e


def test_equal_outcome_but_different_verified_progress_has_action_credit():
    es=[episode(True,seed=100+i*1000,index=i) if i%2==0 else combined_episode(i) for i in range(4)]
    assert not group_advantages(es)['eligible']
    c=compute_credit(es,method=FINE)
    assert c['outcome_zero_variance'] and c['eligible'] and any(d['eligible'] for d in c['depth_statistics'])
    assert c['turn_advantages'][0][3]!=c['turn_advantages'][1][3]


def test_singleton_late_depth_has_zero_credit_and_identical_paths_skip():
    es=[episode(i<3,seed=100+i*1000,index=i) for i in range(4)]
    c=compute_credit(es,method=FINE)
    assert all(c['turn_advantages'][3][t]==0 for t in range(7,12))
    same=[episode(True,seed=100+i*1000,index=i) for i in range(4)]
    assert not compute_credit(same,method=FINE)['eligible']


@pytest.mark.parametrize('scenario',[None,'repeat_sql_error','failed_check_repair'])
def test_fine_shaping_telescopes_and_never_trains_prefix(scenario):
    es=[episode(i%2==0,seed=100+i*1000,index=i,scenario=scenario) for i in range(4)]
    c=compute_credit(es,method=FINE)
    for i,e in enumerate(es):
        expected=sum(r['outcome_reward'] for r in e['turns'])-e['turns'][0]['potential_before']
        assert c['turn_returns'][i][0]==pytest.approx(expected)
        assert len(c['turn_advantages'][i])==len(e['turns'])


def test_fine_advantage_reaches_real_optimizer_gradient(tmp_path):
    import torch
    from apt_v2.grpo_optimizer import response_scores,prepare_group_scores,action_objective,update_group
    model=tiny_model();manual=copy.deepcopy(model);ref=copy.deepcopy(model).requires_grad_(False)
    es=scored_group(model);cache=prepare_group_scores(model,lambda p,r:response_scores(ref,p,r),es,'p','r');c=compute_credit(es,method=FINE)
    total=0.
    for entry in cache['entries']:
        i=entry['episode'];new=response_scores(manual,entry['prompt_token_ids'],entry['response_token_ids'],True)
        old,reference,q=[torch.tensor(entry[k]) for k in ('old','reference','q')]
        loss,_=action_objective(new,old,reference,q,c['turn_advantages'][i][entry['turn']-es[i]['prefix_turns']])
        weight=1/(4*len(es[i]['turns']));(loss*weight).backward();total+=loss.item()*weight
    torch.nn.utils.clip_grad_norm_([p for p in manual.parameters() if p.requires_grad],1.)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=0)
    initial=[p.detach().clone() for p in model.parameters() if p.requires_grad]
    result=update_group(model,opt,es,cache,'p','r',credit_method=FINE)
    assert result['updated'] and result['loss']==pytest.approx(total,abs=1e-7)
    for a,b in zip(model.parameters(),manual.parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=2e-5,atol=1e-6)
    assert any(not torch.equal(a,b) for a,b in zip(initial,[p for p in model.parameters() if p.requires_grad]))
    from grpo_common import atomic_json
    from grpo_c3_common import verify_objective_metrics
    atomic_json(tmp_path/'config.json',{'beta':.02,'clip':.2,'tis_cap':2.,'k':4})
    atomic_json(tmp_path/'groups/group-0000.json',{'episodes':es,'credit':c})
    atomic_json(tmp_path/'checkpoints/group-0000/scores.json',cache)
    atomic_json(tmp_path/'transitions/group-0000.json',{'updated':True,'after':{'checkpoint':'checkpoints/group-0000'},'metrics':result})
    checked=verify_objective_metrics(tmp_path)
    assert checked['passed'] and checked['actual_updates_recomputed']==1 and checked['maximum_metric_difference']<1e-7


@pytest.mark.parametrize('task',closing_tasks(),ids=lambda t:t.id)
def test_actual_sqlite_dependency_removal_and_mutation(task):
    proof=expert_dependency_proof(task);assert proof['passed'] and all(proof['checks'].values())


def test_closing_rejects_skipping_verified_state_and_sql_writes():
    env=ClosingEnvironment(closing_tasks()[0])
    try:
        row=env.step({'tool':'close_period','arguments':{'values':{},'evidence':{},'policy_id':'revenue-v1'}})
        assert row['tool_error'] and env.db.execute('SELECT COUNT(*) FROM period_closings').fetchone()[0]==0
        rows,_=execute_readonly(env.db,next_sql(env.task),TABLES);assert rows==[]
        with pytest.raises(ToolError):execute_readonly(env.db,'DELETE FROM period_closings',TABLES)
        with pytest.raises(ToolError):execute_readonly(env.db,'SELECT * FROM sqlite_master',TABLES)
    finally:env.close()


def test_closing_native_model_path_and_strict_replay():
    from test_dpo_v2 import CharTokenizer
    task=closing_tasks()[0];actions=[r['action'] for r in expert_dependency_proof(task)['expert_records']];tok=CharTokenizer();cursor=[0]
    s={**SAMPLING,'temperature':0,'max_new_tokens':10000,'max_model_len':100000}
    def policy(messages):
        text=json.dumps(actions[cursor[0]]);cursor[0]+=1
        return Generation(text,'greedy',tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True),[ord(c)+1 for c in text]+[0],None,None,'stop')
    e=run_closing(task,policy,s,0,123,tok)
    assert e['success'] and replay_closing(e,tok)==len(actions)
    bad=copy.deepcopy(e);bad['closing_rows'][0][4]+=1
    with pytest.raises(ValueError):replay_closing(bad,tok)


def test_matched_configuration_only_differs_in_credit():
    from grpo_c3_common import matched_configs
    cfg,branches=matched_configs(ROOT/'configs/grpo-c3-matched-v1.json')
    assert cfg['max_wall_seconds']==14400 and branches['fine']['credit_method']==FINE
    from grpo_common import validate_runtime
    actual={'gpu_name':'NVIDIA L40S','gpu_uuid':'NEW-CLONED-L40S','gpu_total_memory':50867666944,'packages':branches['fine']['required_packages']}
    validate_runtime(branches['fine'],actual)
    with pytest.raises(ValueError):validate_runtime(branches['fine'],actual,{**actual,'gpu_uuid':'OLD-L40S'})


def test_next_balance_sql_is_verified_in_perturbed_worlds():
    task=closing_tasks()[0];env=ClosingEnvironment(task)
    try:
        records=expert_dependency_proof(task)['expert_records']
        for row in records:
            env.step(row['action'])
            if row['action']['tool']=='close_period':break
        sql=next_sql(task);rows,reads=execute_readonly(env.db,sql,TABLES)
        assert env._verify_next(sql,rows,reads)=={'next_balance_cents':expected_next(env.db,task)}
        value=expected_next(env.db,task)
        literal=f'SELECT {value} AS next_balance_cents FROM period_closings c JOIN ledger_movements m ON m.region=c.region GROUP BY c.closing_id'
        rows,reads=execute_readonly(env.db,literal,TABLES)
        assert env._verify_next(literal,rows,reads)=={}
        before=env.db.execute('SELECT COUNT(*) FROM period_closings').fetchone()[0]
        closed=env.step(next(r['action'] for r in records if r['action']['tool']=='close_period'))
        assert closed['observation']['already_closed'] and env.db.execute('SELECT COUNT(*) FROM period_closings').fetchone()[0]==before==1
    finally:env.close()


def driver_fixture(tmp_path,monkeypatch,fail_once):
    from types import SimpleNamespace
    import run_v2_grpo_c3 as driver
    from grpo_common import atomic_json,read,file_hash
    root=tmp_path/'v2';cfg={'run_id':'test-matched-c3','max_wall_seconds':14400,'packaging_reserve_seconds':120,'groups_per_branch':2}
    branches={b:{'run_id':'test-'+b,'adapter_sha256':'W','max_groups':2} for b in ['outcome','fine']}
    path=root/'configs/matched.json';atomic_json(path,cfg)
    atomic_json(root/'grpo_c3_release_manifest.json',{'files':{'configs/matched.json':file_hash(path)}})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'matched_configs',lambda p:(cfg,branches))
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/matched.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='NVIDIA L40S, NEW-UUID, driver',stderr=''))
    calls=[];failed=[False]
    def stage(command,**kwargs):
        name=kwargs['log_path'].name.split('-attempt')[0];calls.append(name);kwargs['log_path'].write_text('Progress 100%\r',encoding='utf8')
        target=Path(command[-1])
        if name=='collect-fine-0000' and fail_once and not failed[0]:failed[0]=True;return 1
        if name.startswith('prepare-'):
            atomic_json(target/'input_binding.json',{});atomic_json(target/'execution_environment.json',{'gpu_uuid':'NEW-UUID'})
        if name.startswith('collect-') or name.startswith('refresh-'):
            index=int(command[command.index('--group-index')+1]);atomic_json(target/'groups'/f'group-{index:04}.json',{'index':index})
        if name.startswith('update-'):
            index=int(command[command.index('--group-index')+1]);before=read(Path(command[command.index('--policy-file')+1]))
            atomic_json(target/'transitions'/f'group-{index:04}.json',{'index':index,'before':before,'after':before,'updated':False,'config_sha256':file_hash(target/'config.json'),'group_sha256':file_hash(target/'groups'/f'group-{index:04}.json')})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    return driver,path,calls,root/'artifacts'/cfg['run_id']


def test_matched_driver_alternates_branches_and_evaluates_three_models(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,False)
    archive=driver.run(path)
    assert archive.exists() and [n for n in calls if n.startswith('update-')]==['update-outcome-0000','update-fine-0000','update-outcome-0001','update-fine-0001']
    assert len([n for n in calls if n.startswith('evaluate-')])==3
    assert json.loads((output/'run_status.json').read_text())['committed_groups']=={'outcome':2,'fine':2}


def test_matched_resume_preserves_committed_other_branch_and_budget(tmp_path,monkeypatch):
    driver,path,calls,output=driver_fixture(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError):driver.run(path)
    prior=json.loads((output/'run_status.json').read_text())['elapsed_seconds']
    archive=driver.run(path,True)
    assert archive.name.endswith('attempt-002.zip') and calls.count('update-outcome-0000')==1
    final=json.loads((output/'run_status.json').read_text())
    assert final['error'] is None and final['elapsed_seconds']>=prior and final['max_wall_seconds']==14400


def test_closing_inference_interruption_preserves_partial_evidence():
    calls=[]
    def policy(messages):
        if calls:raise RuntimeError('model service interrupted')
        calls.append(1);return Generation(json.dumps({'tool':'read_policy','arguments':{}}))
    with pytest.raises(RuntimeError) as exc:run_closing(closing_tasks()[0],policy,SAMPLING,0,123)
    evidence=exc.value.episode_evidence
    assert evidence['status']=='interrupted' and not evidence['training_admitted'] and len(evidence['turns'])==1
