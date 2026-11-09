import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import atomic_json,read,file_hash


def fixture(tmp_path,monkeypatch,fail=None,budget_stop=False,zero_control=False):
    import run_v2_grpo_alias_curriculum as driver
    import probe_v2_grpo_alias as probe
    root=tmp_path/'v2';cfg={'run_id':'test-fine','max_wall_seconds':21600,'packaging_reserve_seconds':180,
                         'groups_per_branch':48,'budget_policy':'paired_cycle_with_midpoint_and_final_reservation_v1'}
    branches={b:{'run_id':'test-'+b,'adapter_sha256':'W','max_groups':48} for b in ('control','curriculum')}
    path=root/'configs/study.json';atomic_json(path,cfg)
    atomic_json(root/'grpo_alias_release_manifest.json',{'files':{'configs/study.json':file_hash(path)}})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'matched_configs',lambda p:(cfg,branches))
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/study.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='L40S, UUID, driver',stderr=''))
    def states(child):
        result=[{'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':'W'}]
        for p in sorted((child/'transitions').glob('group-*.json')):result.append(read(p)['after'])
        return result
    monkeypatch.setattr(driver,'states_at',states)
    monkeypatch.setattr(probe,'state_at_midpoint',lambda c:(24,states(c)[24]) if len(states(c))>24 else None)
    if budget_stop:
        monkeypatch.setattr(driver,'training_decision',lambda *a,**k:{'continue':False,'elapsed_seconds':15000})
    calls=[];failed=False
    def stage(command,**kwargs):
        nonlocal failed
        name=kwargs['log_path'].name.split('-attempt')[0];calls.append(name);target=Path(command[-1])
        with kwargs['log_path'].open('x',encoding='utf8') as stream:stream.write('Progress 100%\r')
        assert kwargs['timeout']<=21600-180
        if name==fail and not failed:failed=True;return 1
        if name.startswith('prepare-'):
            atomic_json(target/'input_binding.json',{});atomic_json(target/'execution_environment.json',{'gpu_uuid':'UUID'})
        if name.startswith(('collect-','refresh-')):
            index=int(command[command.index('--group-index')+1]);atomic_json(target/'groups'/f'group-{index:04}.json',{'index':index})
        if name.startswith('update-'):
            index=int(command[command.index('--group-index')+1]);before=read(Path(command[command.index('--policy-file')+1]))
            after={**before,'updates':before['updates']+1,'version':before['version']+1,'checkpoint':f'checkpoints/group-{index:04}'}
            if zero_control and target.name=='control':after=before
            atomic_json(target/'transitions'/f'group-{index:04}.json',{'before':before,'after':after})
        if name.startswith('evaluate-'):
            label=command[command.index('--label')+1];atomic_json(target/'evaluations'/(label+'.json'),{'elapsed_seconds':2100})
        if name.startswith('midpoint-'):atomic_json(target/'diagnostics/midpoint-0024.json',{'spec':read(Path(command[command.index('--spec')+1]))})
        if name=='audit':atomic_json(target/'alias_audit.json',{'passed':True})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    return driver,path,calls,root/'artifacts'/cfg['run_id']


def test_full_driver_alternates_and_binds_both_common_updates(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch);archive=driver.run(path)
    assert archive.exists()
    updates=[n for n in calls if n.startswith('update-')]
    assert updates==[f'update-{b}-{i:04}' for i in range(48) for b in ('control','curriculum')]
    assert len([n for n in calls if n.startswith('evaluate-')])==3
    assert [n for n in calls if n.startswith('refresh-')]==['refresh-control','refresh-curriculum']
    for b in ('control','curriculum'):
        spec=read(output/'branches'/b/'diagnostics/midpoint-0024-policy.json')
        assert spec['group_cursor']==24 and spec['policy']['updates']==24
    assert all(calls.count('midpoint-'+b)==1 for b in ('control','curriculum'))
    assert max(calls.index('midpoint-'+b) for b in ('control','curriculum'))<calls.index('collect-control-0024')
    assert read(output/'training_stop.json')['reason']=='group_cap'
    assert read(output/'run_status.json')['committed_groups']=={'control':48,'curriculum':48}


def test_failure_resume_preserves_commit_and_finishes_orphan_before_stop(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,'update-curriculum-0000',True)
    with pytest.raises(RuntimeError):driver.run(path)
    prior=read(output/'run_status.json')['elapsed_seconds'];archive=driver.run(path,True)
    assert archive.name.endswith('attempt-002.zip') and calls.count('update-control-0000')==1
    status=read(output/'run_status.json');assert status['error'] is None and status['elapsed_seconds']>=prior
    assert status['committed_groups']=={'control':2,'curriculum':2}
    assert read(output/'training_stop.json')['reason']=='budget_reserve'
    assert read(output/'midpoint_status.json')['available_for_both'] is False
    assert len([n for n in calls if n.startswith('evaluate-') and 'grpo_g0002' in n])==2
    assert not [n for n in calls if n.startswith('midpoint-')]


def test_resume_rejects_changed_source_before_new_work(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,'collect-curriculum-0000')
    with pytest.raises(RuntimeError):driver.run(path)
    before=len(calls);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):driver.run(path,True)
    assert len(calls)==before


def test_probe_replays_native_tokens_and_rejects_identity_tampering(tmp_path,monkeypatch):
    import probe_v2_grpo_alias as probe
    import grpo_vllm
    from apt_v2.grpo_alias_curriculum import development_tasks
    from probe_v2_grpo_alias import probe_plan
    from apt_v2.grpo import digest
    from test_grpo_curriculum_v2 import native_episode,NativeTokenizer
    cfg=read(ROOT/'configs/grpo-alias-curriculum-arm-v1.json');tok=NativeTokenizer();tasks={t.id:t for t in development_tasks()}
    adapter=tmp_path/'checkpoints/u12/adapter';adapter.mkdir(parents=True);(adapter/'adapter_model.safetensors').write_bytes(b'fixture-policy')
    spec={'group_cursor':24,'policy':{'checkpoint':'checkpoints/u12','updates':12,'version':12,'adapter_sha256':file_hash(adapter/'adapter_model.safetensors')}}
    atomic_json(tmp_path/'config.json',cfg);atomic_json(tmp_path/'midpoint_plan.json',probe_plan(cfg))
    monkeypatch.setattr(probe,'verify_inputs',lambda *a,**k:cfg);monkeypatch.setattr(probe,'state_at_midpoint',lambda p:(24,spec['policy']))
    tensors={str(i):{'sha256':str(i)} for i in range(112)}
    monkeypatch.setattr(grpo_vllm,'projected_checkpoint_tensors',lambda p:tensors)
    rows=[]
    for p in probe_plan(cfg):
        local=copy.deepcopy(cfg);local['sampling']['temperature']=p['temperature']
        e=native_episode(tasks[p['task_id']],p['scenario'],tok,local,p['seed'],p['rollout_index']);e['policy_version']=12
        e.update(model_branch='midpoint_0024',evaluation_mode=p['mode']);rows.append(e)
    proof={'passed':True,'adapter_sha256':spec['policy']['adapter_sha256'],'registered_ids':[1],'active_slots':[1],
           'tensors':tensors,'loaded_tensor_sha256':digest(tensors)}
    data={'spec':spec,'config_sha256':file_hash(tmp_path/'config.json'),'episodes':rows,'worker_proof':[proof]}
    result=probe.verify_probe(tmp_path,spec,tok,data=data)
    assert result['episodes']==16 and result['summary']['sample']['episodes']==16 and not result['used_for_selection']
    bad=copy.deepcopy(data);bad['episodes'][0]['seed']+=1
    with pytest.raises(ValueError,match='sampling'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    bad=copy.deepcopy(data);bad['episodes'][1]=copy.deepcopy(bad['episodes'][0])
    with pytest.raises(ValueError,match='coverage'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    bad=copy.deepcopy(data);bad['worker_proof'][0]['tensors']['0']['sha256']='tampered'
    with pytest.raises(ValueError,match='tensors'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    (adapter/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):probe.verify_probe(tmp_path,spec,tok,data=data)


def test_midpoint_failure_resume_stays_at_fixed_cursor_before_phase_two(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,'midpoint-curriculum')
    with pytest.raises(RuntimeError):driver.run(path)
    assert read(output/'run_status.json')['committed_groups']=={'control':24,'curriculum':24}
    assert 'collect-control-0024' not in calls
    driver.run(path,True)
    assert calls.count('update-control-0023')==1 and calls.count('update-curriculum-0023')==1
    assert read(output/'branches/control/diagnostics/midpoint-0024-policy.json')['group_cursor']==24
    assert read(output/'run_status.json')['committed_groups']=={'control':48,'curriculum':48}


def test_fixed_midpoint_handles_a_branch_with_zero_actual_updates(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,zero_control=True)
    driver.run(path)
    assert read(output/'run_status.json')['actual_updates']=={'control':0,'curriculum':48}
    spec=read(output/'branches/control/diagnostics/midpoint-0024-policy.json')
    assert spec['group_cursor']==24 and spec['policy']['updates']==0


def test_resume_after_completed_training_never_repeats_midpoint_or_updates(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch)
    driver.run(path)
    protected={str(p):file_hash(p) for b in ('control','curriculum')
               for folder in ('transitions','groups')
               for p in (output/'branches'/b/folder).glob('group-*.json') if p.name!='group-0048.json'}
    for b in ('control','curriculum'):
        child=output/'branches'/b
        (child/'evaluations/grpo_g0048.json').unlink()
        (child/'groups/group-0048.json').unlink()
    (output/'alias_audit.json').unlink()
    status=read(output/'run_status.json')
    status.update(error='FileExistsError: midpoint-control-attempt-001.log',stop_reason='failed')
    atomic_json(output/'run_status.json',status)
    before=len(calls)
    driver.run(path,True)
    resumed=calls[before:]
    assert not any(name.startswith(('collect-','update-','midpoint-')) for name in resumed)
    assert len([name for name in resumed if name.startswith('evaluate-') and 'grpo_g0048' in name])==2
    assert all(file_hash(Path(p))==h for p,h in protected.items())
    result=read(output/'run_status.json')
    assert result['committed_groups']=={'control':48,'curriculum':48} and result['elapsed_seconds']>=status['elapsed_seconds']
