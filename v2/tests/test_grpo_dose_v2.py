import copy,json,sys,shutil
from pathlib import Path
from collections import Counter
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from grpo_common import read,atomic_json,file_hash,validate_config,checkpoint_binding
from apt_v2.grpo_dose_study import *
from test_grpo_repair_study_v2 import result


def master():return read(ROOT/'configs/grpo-sql-dose-study-v1.json')
def branch():return read(ROOT/'configs/grpo-sql-dose-branch-v1.json')

def probe(target=0,maintenance=12,repeat=11):
    from apt_v2.grpo_retention_study import STATES
    return {'target_certifications_by_repeat':[target,target],'maintenance_productive_actions':maintenance,
            'maintenance_by_state':{s:{'contexts':3,'productive':3} for s in STATES},'target_repeat_identical':repeat}


@pytest.mark.parametrize('r',DOSES)
def test_dose_changes_only_target_multiplicity_and_records_correct_ancestry(r):
    from grpo_repair_worker import sft_order,supervision_for,verify_for
    cfg=warm_config(master(),r);data=read(ROOT/cfg['sft_examples']);assert data==supervision_for(cfg)
    assert verify_for(cfg,data);batches=sft_order(cfg);counts=Counter(sum(batches,[]))
    assert len(batches)==12+3*r and len(batches)*4==48+12*r
    assert all(counts[i]==(r if e['role']=='target' else 1) for i,e in enumerate(data))
    assert batches==sft_order(cfg) and cfg['sft']['learning_rate']==2e-6
    c=branch();validate_config(c);c.update(repair_arm='warm',warm_target_repeats=r,source_sft_updates=513+len(batches),adapter_sha256='a'*64)
    validate_config(c)
    c['source_sft_updates']+=1
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('change',[{'max_wall_seconds':14400},{'new_authorized_seconds':0},{'prior_elapsed_seconds':0},
                                  {'target_repeats':[4,16,8]},{'run_grpo':True},{'seed':1}])
def test_new_six_hour_authorization_and_search_order_are_frozen(change):
    c=master();validate_master(c);validate_prior(read(ROOT/c['input_dir']/'prior-evidence.json'),c);c.update(change)
    with pytest.raises(ValueError):validate_master(c)


def test_train_gate_requires_both_repeats_maintenance_stability_and_real_projection_change():
    w=probe();good=probe(target=9,repeat=12)
    assert train_gate(w,good,True)['passed']
    for change in ({'target_certifications_by_repeat':[9,8]},{'maintenance_productive_actions':11},{'target_repeat_identical':10}):
        b=copy.deepcopy(good);b.update(change);assert not train_gate(w,b,True)['passed']
    b=copy.deepcopy(good);b['maintenance_by_state']['stale_evidence_repair']['productive']=2
    assert not train_gate(w,b,True)['passed'] and not train_gate(w,good,False)['passed']


@pytest.mark.parametrize('mode',['train_skip','full_reject_then_pass','all_train_reject','baseline_quality'])
def test_actual_dose_driver_visits_registered_order_and_stops_first_full_pass(tmp_path,monkeypatch,mode):
    import run_v2_grpo_dose as runner
    import grpo_dose_worker as worker
    cfg=master();study=runner.Study(cfg,tmp_path);calls=[]
    w=dict(result(first=0),elapsed_seconds=1900)
    if mode=='baseline_quality':w['evaluation_quality_passed']=False
    def child(name):
        path=tmp_path/'branches'/name;atomic_json(path/'config.json',branch());return path
    def stage(name,*a,**k):
        calls.append(name)
        if name.startswith('gate-'):
            arm=name[5:];passed=mode not in ('all_train_reject','baseline_quality') and (mode!='train_skip' or arm!='r4')
            atomic_json(tmp_path/'train-gates'/(arm+'.json'),{'passed':passed})
    def evaluate(child,name,*a):
        if name=='w':return w
        return result(full=13 if mode=='full_reject_then_pass' and name=='r4' else 14,target=8,first=9)
    monkeypatch.setattr(study,'child',child);monkeypatch.setattr(study,'stage',stage);monkeypatch.setattr(study,'evaluate',evaluate)
    monkeypatch.setattr(worker,'candidate_source',lambda out,name:{'adapter_sha256':'a'*64,'source_sft_updates':513+recipe(int(name[1:]))['steps']})
    answer=study.execute()
    if mode=='baseline_quality':assert answer['stop_reason']=='baseline_quality_rejected' and not any(c.startswith('warm-') for c in calls)
    elif mode=='all_train_reject':assert answer['selected_warm_start'] is None and 'warm-r16' in calls
    else:assert answer['selected_warm_start']=='r8' and 'warm-r16' not in calls and read(tmp_path/'ready-B.json')['dose']==8
    if mode=='train_skip':assert not (tmp_path/'full-gates/r4.json').exists()


@pytest.mark.parametrize('r',DOSES)
def test_prepare_dose_evaluation_accepts_exact_recipe_and_rejects_old_b24_recipe(tmp_path,monkeypatch,r):
    import grpo_common as common
    import apt_v2.grpo_curriculum as curriculum
    c=branch();initial=tmp_path/'artifacts/B';base=tmp_path/'base';base.mkdir()
    (base/'config.json').write_text('{}');(base/'model.safetensors').write_bytes(b'base')
    (initial/'adapter').mkdir(parents=True);(initial/'adapter/adapter_model.safetensors').write_bytes(b'dose B')
    (initial/'adapter/adapter_config.json').write_text('{}');rec=recipe(r);n=rec['steps']
    c.update(repair_arm='warm',warm_target_repeats=r,source_sft_updates=513+n,initial_dir='artifacts/B',base_dir='base',
             adapter_sha256=file_hash(initial/'adapter/adapter_model.safetensors'),profile_input_dir='profile',profile_binding={})
    atomic_json(initial/'complete.json',{'total_sft_updates':513+n,'total_dpo_updates':11,'parent_W_sha256':W_SHA,
                                        'adapter_sha256':{p.name:file_hash(p) for p in (initial/'adapter').iterdir()}})
    atomic_json(initial/'source_audit.json',{'passed':True,'actual_updates':n,'action_exposures':n*4})
    atomic_json(initial/'source_config.json',rec)
    atomic_json(initial/'model_identity.json',{'model_id':c['base_model'],'resolved_revision':c['revision'],
                'files_sha256':{p.name:file_hash(p) for p in base.iterdir()}})
    atomic_json(tmp_path/'profile/audit.json',{'passed':True});atomic_json(tmp_path/'profile/run_status.json',{'error':None})
    out=tmp_path/'prepared';atomic_json(out/'config.json',c);atomic_json(out/'run_manifest.json',{'source_sha256':{}})
    actual={'gpu_name':c.get('required_gpu_name') or 'NVIDIA L40S','gpu_total_memory':c.get('required_min_gpu_bytes',0)+1,'packages':c['required_packages']}
    monkeypatch.setattr(common,'ROOT',tmp_path);monkeypatch.setattr(common,'runtime_identity',lambda:actual)
    monkeypatch.setattr(curriculum,'frozen_inputs',lambda cfg:{'targeted_evaluation_plan.json':[]})
    common.prepare(out);assert read(out/'input_binding.json')['source_sft_updates']==513+n
    atomic_json(initial/'source_config.json',dict(rec,learning_rate=1e-5));atomic_json(tmp_path/'bad/config.json',c)
    with pytest.raises(ValueError,match='recipe'):common.prepare(tmp_path/'bad')


@pytest.mark.parametrize('r',DOSES)
def test_actual_checkpoint_audit_binds_dose_order_optimizer_counters_and_export(tmp_path,monkeypatch,r):
    import torch
    from safetensors.torch import save_file
    import grpo_repair_worker as worker
    from apt_v2.grpo import digest
    from grpo_worker import tensor_digest
    cfg=warm_config(master(),r);data=read(ROOT/cfg['sft_examples']);atomic_json(tmp_path/cfg['sft_examples'],data)
    outer=tmp_path/'main';atomic_json(outer/'config.json',master())
    output=outer/'warmups'/('r'+str(r));atomic_json(output/'config.json',cfg);atomic_json(output/'branches/w/config.json',branch())
    params={f'layer_{i}.lora_A.weight':torch.zeros(2,2) for i in range(112)}
    initial=output/'branches/w/initial/adapter';initial.mkdir(parents=True);save_file(params,str(initial/'adapter_model.safetensors'))
    batches=order(cfg,data);identity=digest({'config':cfg['sft'],'data':data,'order':batches})
    for i in range(cfg['sft']['steps']):
        before=tensor_digest(params);params={k:v+.001 for k,v in params.items()}
        p=output/'warm-start/checkpoints'/f'step-{i+1:04}';(p/'adapter').mkdir(parents=True)
        save_file(params,str(p/'adapter/adapter_model.safetensors'));(p/'adapter/adapter_config.json').write_text('{}')
        atomic_json(p/'metrics.json',{'step':i+1,'indices':batches[i],'optimizer_steps':[i+1],
                                     'frozen_parameter_gradients':0,'losses':[.1]*4,'gradient_norm':.2,
                                     'before_tensor_sha256':before,'after_tensor_sha256':tensor_digest(params),
                                     'adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors')})
        atomic_json(p/'complete.json',{'step':i+1,'identity':identity,'state_sha256':'cloud-state','files':checkpoint_binding(p)})
    b=output/'warm-start/B';shutil.copytree(p/'adapter',b/'adapter');atomic_json(b/'source_config.json',cfg['sft'])
    atomic_json(b/'source_audit.json',{'passed':True,'identity':identity,'train_only':True,'actual_updates':len(batches),'action_exposures':len(batches)*4})
    atomic_json(b/'complete.json',{'adapter_sha256':{p.name:file_hash(p) for p in (b/'adapter').iterdir()},
                'total_sft_updates':513+len(batches),'total_dpo_updates':11,'total_rl_updates':0,'sft_identity':identity,
                'parent_W_sha256':W_SHA,'source_checkpoint':f'checkpoints/step-{len(batches):04}',
                'source_adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors')})
    atomic_json(output/'warm-start/dataset.json',data);monkeypatch.setattr(worker,'ROOT',tmp_path)
    assert worker.audit_sft(output,local=True)['sft_updates']==len(batches)
    import grpo_dose_worker as dose
    monkeypatch.setattr(dose,'ROOT',tmp_path)
    exported=dose.candidate_source(outer,'r'+str(r))
    assert exported['source_sft_updates']==513+len(batches) and exported['adapter_sha256']==file_hash(b/'adapter/adapter_model.safetensors')
    bad=read(p/'metrics.json');bad['indices'][0]=999;atomic_json(p/'metrics.json',bad)
    with pytest.raises(ValueError):worker.audit_sft(output,local=True)


def test_wall_budget_rejection_revokes_ready_warm_start(tmp_path):
    from run_v2_grpo_repair import package
    class Finished:
        cfg={'max_wall_seconds':1};attempt=1
        def elapsed(self):return 2.
    output=tmp_path/'run';output.mkdir();status={'selected_warm_start':'r4','selected_candidate':None,'stop_reason':'warm_start_ready'}
    final=package(output,tmp_path/'review.zip',status,Finished())
    assert final['selected_warm_start'] is None and final['stop_reason']=='wall_budget_exceeded'


def test_new_dose_failure_resume_accounts_packaging_with_separate_prior_budget(tmp_path,monkeypatch):
    import run_v2_grpo_dose as runner
    import run_v2_grpo_repair as base
    c=master();p=tmp_path/'configs/master.json';atomic_json(p,c)
    atomic_json(tmp_path/c['input_dir']/'prior-evidence.json',read(ROOT/c['input_dir']/'prior-evidence.json'))
    atomic_json(tmp_path/runner.MANIFEST,{'files':{'configs/master.json':file_hash(p)}})
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    monkeypatch.setattr(runner,'runtime_identity',lambda:{'sqlite':'3.54.1','gpu_uuid':'fixed'})
    monkeypatch.setattr(base,'resource_snapshot',lambda *a:None)
    monkeypatch.setattr(runner.Study,'execute',lambda self:(_ for _ in ()).throw(RuntimeError('injected dose interruption')))
    with pytest.raises(RuntimeError,match='dose interruption'):runner.run(p)
    receipt=read((tmp_path/(c['run_id']+'-review-attempt-001.zip')).with_suffix('.receipt.json'))
    assert receipt['budget_seconds']==21600 and receipt['prior_elapsed_seconds']==c['prior_elapsed_seconds']
    assert receipt['total_authorized_seconds']==31527.179048960097
    monkeypatch.setattr(runner.Study,'execute',lambda self:{'selected_warm_start':None,'selected_candidate':None,'stop_reason':'no_qualified_warm_start'})
    monkeypatch.setattr(runner.Study,'stage',lambda *a,**k:None)
    runner.run(p,True);second=read((tmp_path/(c['run_id']+'-review-attempt-002.zip')).with_suffix('.receipt.json'))
    assert second['elapsed_seconds_including_packaging']>=receipt['elapsed_seconds_including_packaging']
    with pytest.raises(ValueError,match='completed'):runner.run(p,True)


@pytest.mark.parametrize('qualified',[False,True])
def test_root_audit_recomputes_ordered_search_without_repairing_missing_decisions(tmp_path,monkeypatch,qualified):
    import grpo_dose_worker as audit
    import grpo_repair_worker as sft
    import run_v2_grpo_repair as evaluations
    import run_v2_grpo_dose as runner
    cfg=master();w=result(first=0);b=result(target=8,first=9);records=[]
    hardware={'gpu_uuid':'same'};atomic_json(tmp_path/'execution_environment.json',hardware)
    monkeypatch.setattr(audit,'verify_probe_inputs',lambda *a,**k:(cfg,[]));monkeypatch.setattr(audit,'probe_summary',lambda *a:None)
    monkeypatch.setattr(audit,'branch_inputs',lambda *a,**k:None)
    monkeypatch.setattr(sft,'audit_sft',lambda root,local:{'passed':True,'sft_updates':recipe(int(root.name[1:]))['steps']})
    monkeypatch.setattr(evaluations,'evaluation',lambda child,*a:w if child.name=='w' else b)
    monkeypatch.setattr(runner,'branch_config',lambda *a:{'fixture':'config'})
    monkeypatch.setattr(audit,'candidate_source',lambda output,name:{'adapter_sha256':'a'*64,'source_sft_updates':513+recipe(int(name[1:]))['steps']})
    gates={}
    def gate(output,name,write=True):
        assert write is False
        return gates[name]
    monkeypatch.setattr(audit,'gate',gate)
    for r in DOSES:
        name='r'+str(r);(tmp_path/'warmups'/name).mkdir(parents=True)
        passed=qualified and r==8;gates[name]={'passed':passed};full=b_gate(w,b) if passed else None
        record={'dose':r,'train_gate':gates[name],'full_gate':full,'adapter_sha256':'a'*64};records.append(record)
        atomic_json(tmp_path/'decisions'/(name+'.json'),record)
        if passed:
            atomic_json(tmp_path/'branches'/name/'config.json',{'fixture':'config'})
            atomic_json(tmp_path/'branches'/name/'execution_environment.json',hardware)
            atomic_json(tmp_path/'full-gates'/(name+'.json'),full)
            atomic_json(tmp_path/'ready-B.json',{'adapter_sha256':'a'*64,'source_sft_updates':549,
                       'model_dir':'warmups/r8/warm-start/B','dose':8,'source_dpo_updates':11,'source_rl_updates':0,'parent_W_sha256':W_SHA})
            break
    atomic_json(tmp_path/'selection.json',selection(w,records))
    assert audit.audit(tmp_path,local=True)['passed']
    bad=read(tmp_path/'selection.json');bad['selected_warm_start']='forged';atomic_json(tmp_path/'selection.json',bad)
    with pytest.raises(ValueError,match='selection'):audit.audit(tmp_path,local=True)


def test_counter_chain_reports_only_verified_committed_optimizer_steps(tmp_path):
    from run_v2_grpo_dose import committed_sft_counts
    for i in (1,2):
        p=tmp_path/'warmups/r4/warm-start/checkpoints'/f'step-{i:04}'
        atomic_json(p/'metrics.json',{'optimizer_steps':[i if i==1 else 5]})
        atomic_json(p/'complete.json',{'step':i,'identity':'same','files':{'metrics.json':file_hash(p/'metrics.json')}})
    counts,errors=committed_sft_counts(tmp_path)
    assert counts=={'r4':1} and 'r4' in errors
