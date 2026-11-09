import copy,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import atomic_json,read,file_hash


def fixture(tmp_path,monkeypatch,fail=None,budget_stop=False):
    import run_v2_grpo_fine as driver
    import probe_v2_grpo_fine as probe
    root=tmp_path/'v2';cfg={'run_id':'test-fine','max_wall_seconds':21600,'packaging_reserve_seconds':180,
                         'groups_per_branch':48,'budget_policy':'paired_cycle_with_final_reservation_v1'}
    branches={b:{'run_id':'test-'+b,'adapter_sha256':'W','max_groups':48} for b in ('legacy','local')}
    path=root/'configs/study.json';atomic_json(path,cfg)
    atomic_json(root/'grpo_fine_release_manifest.json',{'files':{'configs/study.json':file_hash(path)}})
    monkeypatch.setattr(driver,'ROOT',root);monkeypatch.setattr(driver,'matched_configs',lambda p:(cfg,branches))
    monkeypatch.setattr(driver,'source_files',lambda:{'configs/study.json':file_hash(path)})
    monkeypatch.setattr(driver,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='L40S, UUID, driver',stderr=''))
    def states(child):
        result=[{'checkpoint':'initial','version':0,'updates':0,'adapter_sha256':'W'}]
        for p in sorted((child/'transitions').glob('group-*.json')):result.append(read(p)['after'])
        return result
    monkeypatch.setattr(driver,'states_at',states)
    monkeypatch.setattr(probe,'state_for_update',lambda c:next(((i,s) for i,s in enumerate(states(c)) if s['updates']==12),None))
    if budget_stop:
        monkeypatch.setattr(driver,'training_decision',lambda *a:{'continue':False,'elapsed_seconds':15000})
    calls=[];failed=False
    def stage(command,**kwargs):
        nonlocal failed
        name=kwargs['log_path'].name.split('-attempt')[0];calls.append(name);target=Path(command[-1])
        kwargs['log_path'].write_text('Progress 100%\r',encoding='utf8')
        assert kwargs['timeout']<=21600-180
        if name==fail and not failed:failed=True;return 1
        if name.startswith('prepare-'):
            atomic_json(target/'input_binding.json',{});atomic_json(target/'execution_environment.json',{'gpu_uuid':'UUID'})
        if name.startswith(('collect-','refresh-')):
            index=int(command[command.index('--group-index')+1]);atomic_json(target/'groups'/f'group-{index:04}.json',{'index':index})
        if name.startswith('update-'):
            index=int(command[command.index('--group-index')+1]);before=read(Path(command[command.index('--policy-file')+1]))
            after={**before,'updates':before['updates']+1,'version':before['version']+1,'checkpoint':f'checkpoints/group-{index:04}'}
            atomic_json(target/'transitions'/f'group-{index:04}.json',{'before':before,'after':after})
        if name.startswith('evaluate-'):
            label=command[command.index('--label')+1];atomic_json(target/'evaluations'/(label+'.json'),{'elapsed_seconds':2100})
        if name.startswith('probe-'):atomic_json(target/'diagnostics/common-update-0012.json',{'spec':read(Path(command[command.index('--spec')+1]))})
        if name=='audit':atomic_json(target/'fine_audit.json',{'passed':True})
        return 0
    monkeypatch.setattr(driver,'run_logged',stage)
    return driver,path,calls,root/'artifacts'/cfg['run_id']


def test_full_driver_alternates_and_binds_both_common_updates(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch);archive=driver.run(path)
    assert archive.exists()
    updates=[n for n in calls if n.startswith('update-')]
    assert updates==[f'update-{b}-{i:04}' for i in range(48) for b in ('legacy','local')]
    assert len([n for n in calls if n.startswith('evaluate-')])==3
    assert [n for n in calls if n.startswith('refresh-')]==['refresh-legacy','refresh-local']
    for b in ('legacy','local'):
        spec=read(output/'branches'/b/'diagnostics/common-update-0012-policy.json')
        assert spec['group_cursor']==12 and spec['policy']['updates']==12
    assert read(output/'training_stop.json')['reason']=='group_cap'
    assert read(output/'run_status.json')['committed_groups']=={'legacy':48,'local':48}


def test_failure_resume_preserves_commit_and_finishes_orphan_before_stop(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,'update-local-0000',True)
    with pytest.raises(RuntimeError):driver.run(path)
    prior=read(output/'run_status.json')['elapsed_seconds'];archive=driver.run(path,True)
    assert archive.name.endswith('attempt-002.zip') and calls.count('update-legacy-0000')==1
    status=read(output/'run_status.json');assert status['error'] is None and status['elapsed_seconds']>=prior
    assert status['committed_groups']=={'legacy':2,'local':2}
    assert read(output/'training_stop.json')['reason']=='budget_reserve'
    assert read(output/'common_update_probe_status.json')['available_for_both'] is False
    assert len([n for n in calls if n.startswith('evaluate-') and 'grpo_g0002' in n])==2
    assert not [n for n in calls if n.startswith('probe-')]


def test_resume_rejects_changed_source_before_new_work(tmp_path,monkeypatch):
    driver,path,calls,output=fixture(tmp_path,monkeypatch,'collect-local-0000')
    with pytest.raises(RuntimeError):driver.run(path)
    before=len(calls);path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError):driver.run(path,True)
    assert len(calls)==before


def test_probe_replays_native_tokens_and_rejects_identity_tampering(tmp_path,monkeypatch):
    import probe_v2_grpo_fine as probe
    import grpo_vllm
    from apt_v2.grpo_fine_study import development_tasks,probe_plan
    from apt_v2.grpo import digest
    from test_grpo_curriculum_v2 import native_episode,NativeTokenizer
    cfg=read(ROOT/'configs/grpo-fine-credit-local-v1.json');tok=NativeTokenizer();tasks={t.id:t for t in development_tasks()}
    adapter=tmp_path/'checkpoints/u12/adapter';adapter.mkdir(parents=True);(adapter/'adapter_model.safetensors').write_bytes(b'fixture-policy')
    spec={'group_cursor':15,'actual_updates':12,'policy':{'checkpoint':'checkpoints/u12','updates':12,'version':12,'adapter_sha256':file_hash(adapter/'adapter_model.safetensors')}}
    atomic_json(tmp_path/'config.json',cfg);atomic_json(tmp_path/'common_update_probe_plan.json',probe_plan(cfg))
    monkeypatch.setattr(probe,'verify_inputs',lambda *a,**k:cfg);monkeypatch.setattr(probe,'state_for_update',lambda p:(15,spec['policy']))
    tensors={str(i):{'sha256':str(i)} for i in range(112)}
    monkeypatch.setattr(grpo_vllm,'projected_checkpoint_tensors',lambda p:tensors)
    rows=[]
    for p in probe_plan(cfg):
        local=copy.deepcopy(cfg);local['sampling']['temperature']=p['temperature']
        e=native_episode(tasks[p['task_id']],p['scenario'],tok,local,p['seed'],p['rollout_index']);e['policy_version']=12
        e.update(model_branch='common_update_0012',evaluation_mode=p['mode']);rows.append(e)
    proof={'passed':True,'adapter_sha256':spec['policy']['adapter_sha256'],'registered_ids':[1],'active_slots':[1],
           'tensors':tensors,'loaded_tensor_sha256':digest(tensors)}
    data={'spec':spec,'config_sha256':file_hash(tmp_path/'config.json'),'episodes':rows,'worker_proof':[proof]}
    result=probe.verify_probe(tmp_path,spec,tok,data=data)
    assert result['episodes']==16 and result['by_kind']['targeted']['episodes']==12 and not result['used_for_selection']
    bad=copy.deepcopy(data);bad['episodes'][0]['seed']+=1
    with pytest.raises(ValueError,match='sampling'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    bad=copy.deepcopy(data);bad['episodes'][1]=copy.deepcopy(bad['episodes'][0])
    with pytest.raises(ValueError,match='coverage'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    bad=copy.deepcopy(data);bad['worker_proof'][0]['tensors']['0']['sha256']='tampered'
    with pytest.raises(ValueError,match='tensors'):probe.verify_probe(tmp_path,spec,tok,data=bad)
    (adapter/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):probe.verify_probe(tmp_path,spec,tok,data=data)
