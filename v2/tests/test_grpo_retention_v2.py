import copy,json,sys,shutil
from collections import Counter
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,file_hash,validate_config,checkpoint_binding
from apt_v2.grpo_retention_study import *
from test_grpo_repair_study_v2 import result


def master():return read(ROOT/'configs/grpo-sql-retention-study-v1.json')
def branch():return read(ROOT/'configs/grpo-sql-retention-branch-v1.json')


def test_all_twelve_train_tasks_have_target_and_four_real_recovery_actions():
    from grpo_repair_worker import verify_supervision,supervision_for,sft_order
    cfg=master();data=supervision_dataset();assert data==supervision_for(cfg)
    assert len(data)==60 and Counter(e['state'] for e in data)==Counter({s:12 for s in (COMPOUND,*STATES)})
    assert Counter(e['task_id'] for e in data)==Counter({t.id:5 for t in training_tasks()})
    assert verify_supervision(data,expected_count=60)
    for e in data:
        if e['state']=='checked_submit':
            obs=json.loads(e['messages'][-1]['content'])['observation']
            assert obs['passed'] and not obs['submitted'] and e['observation']['submitted']
    batches=sft_order(cfg);assert len(batches)==15 and sorted(sum(batches,[]))==list(range(60))
    assert batches==sft_order(cfg)
    bad=copy.deepcopy(data);bad[0]['messages'][-1]['content']='{"observation": {"error":"fake"}}'
    with pytest.raises(ValueError):verify_supervision(bad,expected_count=60)
    bad=copy.deepcopy(data);bad[-1]['task_id']=bad[0]['task_id']
    with pytest.raises(ValueError):verify_dataset(bad)


@pytest.mark.parametrize('change',[{'max_wall_seconds':16001},{'prior_elapsed_seconds':0},{'sft':dict(RECIPE,steps=24)},
                                   {'sft':dict(RECIPE,learning_rate=1e-5)},{'route':'online_plain'}])
def test_master_rejects_budget_recipe_or_route_drift(change):
    cfg=master();validate_master(cfg);cfg.update(change)
    with pytest.raises(ValueError):validate_master(cfg)


def test_prior_route_is_bound_and_legacy_recipe_is_unchanged():
    from grpo_repair_worker import sft_order,supervision_dataset as legacy
    cfg=master();prior=read(ROOT/cfg['prior_evidence']);assert validate_prior(prior,cfg)
    bad=copy.deepcopy(prior);bad['receipt']['elapsed_seconds_including_packaging']=0
    with pytest.raises(ValueError):validate_prior(bad,cfg)
    old=read(ROOT/'configs/grpo-sql-repair-study-v1.json')
    assert len(legacy())==24 and len(sft_order(old))==24 and old['sft']['learning_rate']==1e-5


def test_warm_branch_is_sft528_and_rejects_legacy_or_unknown_ancestry():
    cfg=branch();validate_config(cfg)
    cfg.update(repair_arm='warm',source_sft_updates=528,adapter_sha256='a'*64)
    validate_config(cfg)
    for change in ({'source_sft_updates':537},{'repair_profile':'invented'},{'run_id':'old-study-identifier'}):
        bad=copy.deepcopy(cfg);bad.update(change)
        with pytest.raises(ValueError):validate_config(bad)


@pytest.mark.parametrize('outcome',['pass','reject','baseline_quality'])
def test_retention_driver_gates_b_before_grpo_without_rerunning_diagnostic(tmp_path,monkeypatch,outcome):
    import run_v2_grpo_retention as runner
    cfg=master();study=runner.Study(cfg,tmp_path);calls=[]
    w=dict(result(),elapsed_seconds=1900);b=dict(result(full=15,target=4,first=5),elapsed_seconds=2000)
    if outcome=='reject':b=result(full=13,target=8,first=8)
    if outcome=='baseline_quality':w['evaluation_quality_passed']=False
    def child(name):
        p=tmp_path/'branches'/name;atomic_json(p/'config.json',branch());return p
    monkeypatch.setattr(study,'child',child);monkeypatch.setattr(study,'stage',lambda name,*a,**k:calls.append(name))
    monkeypatch.setattr(study,'evaluate',lambda c,n,*a:b if n=='b' else w)
    def train(names,baseline):
        assert outcome=='pass' and names==['c']
        calls.append('train-c');return {'c':result(full=16,target=5,first=6)},True
    monkeypatch.setattr(study,'train',train)
    answer=study.execute()
    assert 'diagnostic' not in calls
    if outcome=='pass':assert answer['selected_candidate']=='c'
    elif outcome=='reject':assert answer['stop_reason']=='b_gate_rejected' and 'train-c' not in calls
    else:assert answer['stop_reason']=='baseline_quality_rejected' and 'warm-start' not in calls


def test_retention_failure_package_resume_and_combined_budget_receipt(tmp_path,monkeypatch):
    import run_v2_grpo_retention as runner
    import run_v2_grpo_repair as base
    cfg=master();p=tmp_path/'configs/master.json';atomic_json(p,cfg)
    atomic_json(tmp_path/cfg['prior_evidence'],read(ROOT/cfg['prior_evidence']))
    atomic_json(tmp_path/runner.MANIFEST,{'files':{'configs/master.json':file_hash(p),cfg['prior_evidence']:file_hash(tmp_path/cfg['prior_evidence'])}})
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'sqlite':'3.54.1','gpu_uuid':'same'})
    monkeypatch.setattr(base,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(runner.Study,'execute',lambda self:(_ for _ in ()).throw(RuntimeError('injected retention interruption')))
    with pytest.raises(RuntimeError,match='injected retention'):runner.run(p)
    archive=tmp_path/(cfg['run_id']+'-review-attempt-001.zip');receipt=read(archive.with_suffix('.receipt.json'))
    assert receipt['budget_seconds']==16000 and receipt['prior_elapsed_seconds']==cfg['prior_elapsed_seconds']
    assert receipt['combined_elapsed_seconds']==cfg['prior_elapsed_seconds']+receipt['elapsed_seconds_including_packaging']
    monkeypatch.setattr(runner.Study,'execute',lambda self:{'selected_candidate':None,'stop_reason':'completed'})
    monkeypatch.setattr(runner.Study,'stage',lambda *a,**k:None)
    runner.run(p,True);new=read((tmp_path/(cfg['run_id']+'-review-attempt-002.zip')).with_suffix('.receipt.json'))
    assert new['elapsed_seconds_including_packaging']>=receipt['elapsed_seconds_including_packaging']
    with pytest.raises(ValueError,match='completed'):runner.run(p,True)


def test_prepare_new_c_binds_only_fifteen_step_sixty_exposure_b(tmp_path,monkeypatch):
    import grpo_common as common
    import apt_v2.grpo_curriculum as curriculum
    c=branch();initial=tmp_path/'artifacts/B';base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{}');(base/'model.safetensors').write_bytes(b'base')
    (initial/'adapter').mkdir(parents=True);(initial/'adapter/adapter_model.safetensors').write_bytes(b'new B')
    (initial/'adapter/adapter_config.json').write_text('{}')
    c.update(repair_arm='warm',source_sft_updates=528,initial_dir='artifacts/B',base_dir='base',
             adapter_sha256=file_hash(initial/'adapter/adapter_model.safetensors'),profile_input_dir='profile',profile_binding={})
    atomic_json(initial/'complete.json',{'total_sft_updates':528,'total_dpo_updates':11,'parent_W_sha256':W_SHA,
                                        'adapter_sha256':{p.name:file_hash(p) for p in (initial/'adapter').iterdir()}})
    atomic_json(initial/'source_audit.json',{'passed':True,'actual_updates':15,'action_exposures':60})
    atomic_json(initial/'model_identity.json',{'model_id':c['base_model'],'resolved_revision':c['revision'],
                'files_sha256':{p.name:file_hash(p) for p in base.iterdir()}})
    atomic_json(tmp_path/'profile/audit.json',{'passed':True});atomic_json(tmp_path/'profile/run_status.json',{'error':None})
    out=tmp_path/'prepared';atomic_json(out/'config.json',c);atomic_json(out/'run_manifest.json',{'source_sha256':{}})
    actual={'gpu_name':c.get('required_gpu_name') or 'NVIDIA L40S','gpu_total_memory':c.get('required_min_gpu_bytes',0)+1,'packages':c['required_packages']}
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'runtime_identity',lambda:actual)
    monkeypatch.setattr(curriculum,'frozen_inputs',lambda cfg:{'targeted_evaluation_plan.json':[]})
    common.prepare(out);assert read(out/'input_binding.json')['source_sft_updates']==528
    atomic_json(initial/'source_audit.json',{'passed':True,'actual_updates':24,'action_exposures':96})
    atomic_json(tmp_path/'bad/config.json',c)
    with pytest.raises(ValueError,match='ancestry'):common.prepare(tmp_path/'bad')


def test_actual_sft_checkpoint_audit_accepts_15_steps_and_rejects_metric_tampering(tmp_path,monkeypatch):
    import torch
    from safetensors.torch import save_file
    import grpo_repair_worker as worker
    from apt_v2.grpo import digest
    from grpo_worker import tensor_digest
    cfg=master();data=supervision_dataset();atomic_json(tmp_path/cfg['sft_examples'],data)
    output=tmp_path/'run';atomic_json(output/'config.json',cfg)
    atomic_json(output/'branches/w/config.json',branch())
    params={f'layer_{i}.lora_A.weight':torch.zeros(2,2) for i in range(112)}
    initial=output/'branches/w/initial/adapter';initial.mkdir(parents=True)
    save_file(params,str(initial/'adapter_model.safetensors'))
    identity=digest({'config':cfg['sft'],'data':data,'order':order(cfg)})
    for i in range(15):
        before=tensor_digest(params);params={k:v+.001 for k,v in params.items()}
        p=output/'warm-start/checkpoints'/f'step-{i+1:04}';(p/'adapter').mkdir(parents=True)
        save_file(params,str(p/'adapter/adapter_model.safetensors'));(p/'adapter/adapter_config.json').write_text('{}')
        atomic_json(p/'metrics.json',{'step':i+1,'indices':order(cfg)[i],'optimizer_steps':[i+1],
                                     'frozen_parameter_gradients':0,'losses':[.1]*4,'gradient_norm':.2,
                                     'before_tensor_sha256':before,'after_tensor_sha256':tensor_digest(params),
                                     'adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors')})
        torch.save({'identity':identity,'step':i+1,'optimizer':{'state':{0:{'step':torch.tensor(i+1)}}}},p/'state.pt')
        atomic_json(p/'complete.json',{'step':i+1,'identity':identity,'state_sha256':file_hash(p/'state.pt'),
                                      'files':checkpoint_binding(p)})
    b=output/'warm-start/B';shutil.copytree(p/'adapter',b/'adapter')
    atomic_json(b/'source_config.json',cfg['sft'])
    atomic_json(b/'source_audit.json',{'passed':True,'identity':identity,'train_only':True,'actual_updates':15,'action_exposures':60})
    atomic_json(b/'complete.json',{'adapter_sha256':{p.name:file_hash(p) for p in (b/'adapter').iterdir()},
                'total_sft_updates':528,'total_dpo_updates':11,'total_rl_updates':0,'sft_identity':identity,
                'parent_W_sha256':W_SHA,'source_checkpoint':'checkpoints/step-0015',
                'source_adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors')})
    atomic_json(output/'warm-start/dataset.json',data);monkeypatch.setattr(worker,'ROOT',tmp_path)
    assert worker.audit_sft(output,local=True)['sft_updates']==15
    assert worker.audit_sft(output,local=False)['action_exposures']==60
    bad=read(p/'metrics.json');bad['optimizer_steps']=[24];atomic_json(p/'metrics.json',bad)
    with pytest.raises(ValueError):worker.audit_sft(output,local=True)


@pytest.mark.parametrize('passed',[False,True])
def test_root_audit_recomputes_b_gate_and_optional_c_selection(tmp_path,monkeypatch,passed):
    import audit_v2_grpo_retention as audit
    import audit_v2_grpo as branch_audit
    import grpo_c3_common as objective
    from run_v2_grpo_repair import select
    from apt_v2.grpo_repair_study import b_gate
    cfg=master();prior=read(ROOT/cfg['prior_evidence']);atomic_json(tmp_path/cfg['prior_evidence'],prior)
    out=tmp_path/'run';atomic_json(out/'config.json',cfg);atomic_json(out/'prior-evidence.json',prior)
    atomic_json(out/'release_manifest.json',{'files':{}})
    atomic_json(out/'run_manifest.json',{'source_sha256':{},'release_sha256':file_hash(out/'release_manifest.json')})
    atomic_json(out/'execution_environment.json',{'same':'hardware'})
    for name in ('w','c'):atomic_json(out/'branches'/name/'execution_environment.json',{'same':'hardware'})
    w=dict(result(),elapsed_seconds=1900);b=result(full=15 if passed else 13,target=4,first=5)
    c=result(full=16,target=5,first=6);gate=b_gate(w,b);atomic_json(out/'warm-start/b-gate.json',gate)
    expected=({'route':'warm_start','stop_reason':'completed',**select(w,{'c':c},b,True)} if passed else
              {'route':'warm_start','stop_reason':'b_gate_rejected','selected_candidate':None,'b_gate':gate})
    atomic_json(out/'selection.json',expected)
    monkeypatch.setattr(audit,'ROOT',tmp_path);monkeypatch.setattr(audit,'verify_inputs',lambda *a,**k:None)
    monkeypatch.setattr(audit,'audit_sft',lambda *a,**k:{'passed':True,'sft_updates':15})
    monkeypatch.setattr(audit,'evaluation',lambda child,label,local:w if child.name=='w' else b if label=='weighted_sft' else c)
    monkeypatch.setattr(audit,'states',lambda child:(36,{'updates':1}))
    monkeypatch.setattr(branch_audit,'audit',lambda *a,**k:{'passed':True,'effective_bf16_tensor_hash_changed':True})
    monkeypatch.setattr(objective,'verify_objective_metrics',lambda *a:{'passed':True})
    assert audit.audit(out,local=True)['passed']
    bad=copy.deepcopy(expected);bad['selected_candidate']='fabricated';atomic_json(out/'selection.json',bad)
    with pytest.raises(ValueError,match='selection'):audit.audit(out,local=True)
