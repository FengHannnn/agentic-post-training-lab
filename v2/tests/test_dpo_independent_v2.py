import sys,json,copy,hashlib
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_dpo_v2 import rollout,CharTokenizer
from apt_v2.tasks import make_tasks
from apt_v2.dpo_test import replay_episode,validate_set,paired_outcomes
from apt_v2.sft_diagnostics import summarize_episodes
from common import file_hash

def test_holdout_native_replay_detects_token_and_environment_tamper():
    task=make_tasks()[-4];episode,tok=rollout(task)
    assert replay_episode(task,episode,tok,'dpo_sample')==len(episode['turns'])
    bad=copy.deepcopy(episode);bad['turns'][0]['generation']['prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='prompt'):replay_episode(task,bad,tok,'dpo_sample')
    bad=copy.deepcopy(episode);bad['turns'][0]['reward']+=1
    with pytest.raises(ValueError,match='reward'):replay_episode(task,bad,tok,'dpo_sample')
    bad=copy.deepcopy(episode);bad['family_id']='train-family'
    with pytest.raises(ValueError,match='identity'):replay_episode(task,bad,tok,'dpo_sample')

def fixtures(tasks,branch,sc=None):
    episodes=[]
    for ti,task in enumerate(tasks):
        for ri in range(sc['rollouts_per_task'] if sc else 1):
            e,_=rollout(task);e['model_branch']=branch
            if sc:
                seed=sc['seed']+ti*10000+ri*100;e.update(rollout_index=ri,sample_seed=seed,temperature=sc['temperature'])
                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
            else:
                for row in e['turns']:row['generation']['mode']='greedy'
            episodes.append(e)
    return episodes

def test_holdout_complete_sets_seed_binding_and_paired_outcomes():
    tasks=make_tasks()[-4:];sc={'seed':17,'rollouts_per_task':2,'temperature':.8}
    episodes=fixtures(tasks,'sft',sc);validate_set(episodes,tasks,'sft',sc)
    with pytest.raises(ValueError,match='distinct'):validate_set(episodes[:-1],tasks,'sft',sc)
    bad=copy.deepcopy(episodes);bad[0]['turns'][0]['sampling_seed']+=1
    with pytest.raises(ValueError,match='seed'):validate_set(bad,tasks,'sft',sc)
    bad=copy.deepcopy(episodes);bad[0]['split']='train'
    with pytest.raises(ValueError,match='branch'):validate_set(bad,tasks,'sft',sc)
    other=copy.deepcopy(episodes);other[0]['success']=False
    assert paired_outcomes(episodes,other,True)['regressed']==1
    with pytest.raises(ValueError,match='identities'):paired_outcomes(episodes,other[:-1],True)

def test_independent_audit_replays_complete_72_episodes(tmp_path,monkeypatch):
    import test_v2_dpo as worker
    import transformers
    tasks=make_tasks()[-4:];sc={'seed':19,'rollouts_per_task':8,'temperature':.8}
    cfg={'sample_eval':sc};monkeypatch.setattr(worker,'verify_inputs',lambda output:cfg)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:CharTokenizer())
    (tmp_path/'model_identity.json').write_text('{"files_sha256":{}}')
    comparison={};stochastic={}
    for branch in ('sft','dpo'):
        for mode in ('greedy','sample'):
            episodes=fixtures(tasks,branch,sc if mode=='sample' else None)
            name=branch+('-sample' if mode=='sample' else '-test')
            path=tmp_path/(name+'-episodes.jsonl');path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)}
            if mode=='sample':summary['successes_by_task']={t.id:8 for t in tasks};stochastic[branch]=summary
            else:comparison[branch]={'test':summary}
            (tmp_path/(name+'-summary.json')).write_text(json.dumps(summary))
    (tmp_path/'comparison_summary.json').write_text(json.dumps(comparison));(tmp_path/'stochastic_comparison.json').write_text(json.dumps(stochastic))
    worker.audit(tmp_path)
    assert json.loads((tmp_path/'audit.json').read_text())['episodes_replayed']==72
    result=json.loads((tmp_path/'independent_test_summary.json').read_text())
    assert result['paired_sample']['both_success']==32 and result['optimizer_updates']==0
    stochastic['dpo']['successes']=31;(tmp_path/'stochastic_comparison.json').write_text(json.dumps(stochastic))
    with pytest.raises(ValueError,match='summary'):worker.audit(tmp_path)

def test_prepare_pins_adapters_and_train_only_provenance(tmp_path,monkeypatch):
    import test_v2_dpo as worker
    from apt_v2.sql_sft import data_identity
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    put=lambda path,value:(path.parent.mkdir(parents=True,exist_ok=True),path.write_text(json.dumps(value)))
    cfg={'parent_run':'parent','candidate_run':'candidate','data_run':'data','training_enabled':False,'eval':{'splits':{'test':4}},'sample_eval':{'task_count':4,'rollouts_per_task':8}}
    for branch in ('parent','candidate'):
        folder=tmp_path/'artifacts'/branch;adapter=folder/'final/adapter/adapter_model.safetensors'
        adapter.parent.mkdir(parents=True);adapter.write_bytes(branch.encode())
        key='adapter_sha256' if branch=='parent' else 'candidate_adapter_sha256';cfg[key]=file_hash(adapter)
        complete={'adapter_sha256':{'adapter_model.safetensors':file_hash(adapter)}}
        if branch=='parent':complete['total_steps']=126
        else:complete.update(algorithm='dpo',steps=11)
        put(folder/'final/complete.json',complete)
        put(folder/'model_identity.json',{'resolved_revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','files_sha256':{}})
    train=make_tasks()[0];examples=[{'task_id':train.id,'family_id':train.family_id,'split':'train'}]
    parent=tmp_path/'artifacts/parent';(parent/'examples.jsonl').write_text(json.dumps(examples[0])+'\n')
    put(parent/'input_binding.json',{'dataset_sha256':data_identity(examples)})
    pairs=''.join(json.dumps(examples[0])+'\n' for _ in range(44))
    for branch in ('candidate','data'):
        folder=tmp_path/'artifacts'/branch;folder.mkdir(exist_ok=True);(folder/'pairs.jsonl').write_text(pairs)
    cfg['pairs_sha256']=file_hash(tmp_path/'artifacts/candidate/pairs.jsonl')
    put(tmp_path/'artifacts/candidate/config.json',{'adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256']})
    put(tmp_path/'artifacts/candidate/training_summary.json',{'optimizer_updates':11})
    put(tmp_path/'configs/dpo-independent-test-v1.json',cfg)
    output=tmp_path/'output';put(output/'config.json',cfg);put(output/'run_manifest.json',{'source_sha256':{}})
    worker.prepare(output)
    assert len(json.loads((output/'test_tasks.json').read_text()))==4
    assert json.loads((output/'prepare_summary.json').read_text())['sample_episodes']==64
    (tmp_path/'artifacts/candidate/final/adapter/adapter_model.safetensors').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='identity'):worker.verify_inputs(output)

def test_independent_runner_failure_preserves_logs_and_zero_updates(tmp_path,monkeypatch):
    import run_v2_dpo_test as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir();candidate=tmp_path/'artifacts/candidate'
    for name in ('pairs.jsonl','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        path=candidate/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    (tmp_path/'configs/dpo-independent-test-v1.json').write_text(json.dumps({'run_id':'test-run','candidate_run':'candidate','max_wall_seconds':60}))
    (tmp_path/'dpo_test_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/test_v2_dpo.py').write_text('raise RuntimeError("intentional")')
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/test-run/run_status.json').read_text())
    assert status['optimizer_updates']==0 and status['training_enabled'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'test-run-review.zip') as z:assert 'test-run/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()
