from pathlib import Path
import json,hashlib,sys,importlib.util
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-independent-test-v1-20261102/extracted/dpo-independent-test-v1-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-independent-test-v1-20261102')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/dpo-independent-test-v1.json')
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
runs={'sql-sft-weighted-v1-20261101':ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101',
'dpo-train-v1-20261102':ROOT/'artifacts/reviews/dpo-train-v1-20261102/extracted/dpo-train-v1-20261102',
'dpo-collect-v1-20261102':ROOT/'artifacts/reviews/dpo-collect-v1-20261102/extracted/dpo-collect-v1-20261102'}
for branch,run in [('sft',cfg['parent_run']),('dpo',cfg['candidate_run'])]:
 for n,h in binding['checkpoints'][branch].items():assert sha(runs[run]/'final'/n)==h,(branch,n)
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts;assert sha(runs[parts[1]].joinpath(*parts[2:]))==h,rel
for n,h in binding['checkpoints']['dpo'].items():
 if (out/'final'/n).exists():assert sha(out/'final'/n)==h,n
identity=read(out/'model_identity.json');assert identity==read(runs[cfg['parent_run']]/'model_identity.json')
assert identity['files_sha256']==binding['model_files']
for n,h in binding['model_files'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('reviewed_dpo_test',out/'source/scripts/test_v2_dpo.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from dataclasses import asdict
from apt_v2.tasks import make_tasks
from apt_v2.sft_test import holdout_tasks
from apt_v2.sql_sft import data_identity
examples=[json.loads(l) for l in (runs[cfg['parent_run']]/'examples.jsonl').read_text(encoding='utf8').splitlines()]
assert data_identity(examples)==read(runs[cfg['parent_run']]/'input_binding.json')['dataset_sha256']
tasks=holdout_tasks(cfg,examples)
assert [asdict(t) for t in tasks]==read(out/'test_tasks.json')
allowed={(t.id,t.family_id) for t in make_tasks() if t.split=='train'}
pairs=[json.loads(l) for l in (runs[cfg['candidate_run']]/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
assert len(pairs)==44 and all(p['split']=='train' and (p['task_id'],p['family_id']) in allowed for p in pairs)
worker.verify_inputs=lambda output:cfg
worker.write=lambda path,value:(dest/('local_'+path.name)).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
worker.audit(out)
assert read(dest/'local_audit.json')==read(out/'audit.json')
assert read(dest/'local_independent_test_summary.json')==read(out/'independent_test_summary.json')
result={'zip_sha256':sha(dest/'dpo-independent-test-v1-20261102-review.zip'),'source_files_verified':len(sources),'source_matches_local':True,
'checkpoint_files_verified':sum(len(v) for v in binding['checkpoints'].values()),'provenance_files_verified':len(binding['provenance']),
'model_identity_matches_reviewed_baseline':True,'tokenizer_files_verified':sum((out/'tokenizer'/n).exists() for n in binding['model_files']),
'episodes_replayed':read(out/'audit.json')['episodes_replayed'],'turns_replayed':read(out/'audit.json')['turns_replayed'],
'local_audit_matches_cloud':True,'local_summary_matches_cloud':True,'unique_test_tasks':len(tasks),'training_enabled':False,'optimizer_updates':0}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
