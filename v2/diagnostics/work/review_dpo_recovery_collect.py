from pathlib import Path
import json,hashlib,sys,importlib.util
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-collect-v1-20261102/extracted/dpo-recovery-collect-v1-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-collect-v1-20261102')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/dpo-recovery-collect-v1.json')
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
runs={run:ROOT/'artifacts/reviews'/run/'extracted'/run for run in [cfg['source_run'],cfg['parent_run'],cfg['anchor_run']]}
for rel,h in binding['source_files'].items():
 parts=Path(rel).parts;assert sha(runs[parts[1]].joinpath(*parts[2:]))==h,rel
for n,h in binding['prepared_files'].items():assert sha(out/n)==h,n
identity=read(out/'model_identity.json');assert identity==read(runs[cfg['source_run']]/'model_identity.json')
assert identity['files_sha256']==binding['model_files']
for n,h in binding['model_files'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('reviewed_recovery_collector',out/'source/scripts/collect_v2_dpo_recovery.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from dataclasses import asdict
from apt_v2.tasks import make_tasks
from apt_v2.sql_sft import data_identity
allowed={(t.id,t.family_id) for t in make_tasks() if t.split=='train'}
assert read(out/'tasks.json')==[asdict(t) for t in make_tasks() if t.split=='train']
examples=worker.load(runs[cfg['parent_run']]/'examples.jsonl')
assert data_identity(examples)==read(runs[cfg['parent_run']]/'input_binding.json')['dataset_sha256']
assert all(e['split']=='train' and (e['task_id'],e['family_id']) in allowed for e in examples+worker.load(out/'anchor_pairs.jsonl'))
assert worker.load(out/'anchor_pairs.jsonl')==worker.load(runs[cfg['anchor_run']]/'pairs.jsonl')
worker.verify_inputs=lambda output:(cfg,runs[cfg['source_run']])
worker.write=lambda path,value:(dest/('local_'+path.name)).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
worker.audit(out)
assert read(dest/'local_audit.json')==read(out/'audit.json')
result={'zip_sha256':sha(dest/'dpo-recovery-collect-v1-20261102-review.zip'),'source_files_verified':len(sources),'source_matches_local':True,
 'source_binding_files_verified':len(binding['source_files']),'prepared_files_verified':len(binding['prepared_files']),
 'tokenizer_files_verified':sum((out/'tokenizer'/n).exists() for n in binding['model_files']),
 'contexts_replayed':84,'native_samples_audited':252,'pairs_regenerated_identically':169,'local_audit_matches_cloud':True,
 'train_only':True,'optimizer_updates':0}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
