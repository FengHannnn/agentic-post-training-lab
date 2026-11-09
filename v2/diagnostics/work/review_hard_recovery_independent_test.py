# Use bundled SQLite 3.53.1 to match recorded syntax-error locations; no source/data mutation.
import ctypes
_sqlite_runtime=ctypes.WinDLL(r'C:/Users/app/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/DLLs/sqlite3.dll')
import sqlite3
from pathlib import Path
import json,hashlib,sys,importlib.util,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='hard-recovery-independent-test-v1-20261103';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/hard-recovery-independent-test-v1.json')
release=read(ROOT/'hard_recovery_test_release_manifest.json')['files']
for n,h in sources.items():
 assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
 assert release.get('v2/'+n,release.get(n))==h,n
for n,k in [('config.json','config_sha256'),('pairs.jsonl','pairs_sha256'),('examples.jsonl','examples_file_sha256'),('test_tasks.json','test_tasks_sha256'),('contexts.jsonl','contexts_sha256')]:assert sha(out/n)==binding[k],n
assert sha(out/'pairs.jsonl')==cfg['pairs_sha256']
assert binding['training_enabled'] is False and binding['optimizer_updates']==0
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts;prior=ROOT/'artifacts/reviews'/parts[1]/'extracted'/parts[1]
 assert sha(prior.joinpath(*parts[2:]))==h,rel
for branch,key,modelkey in [('before','parent_run','adapter_sha256'),('after','candidate_run','candidate_adapter_sha256')]:
 prior=ROOT/'artifacts/reviews'/cfg[key]/'extracted'/cfg[key]/'final'
 for n,h in binding['checkpoints'][branch].items():assert sha(prior/n)==h,(branch,n)
 assert sha(prior/'adapter/adapter_model.safetensors')==cfg[modelkey]
identity=read(out/'model_identity.json');assert identity['files_sha256']==binding['model_files']
for n,h in identity['files_sha256'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
parent=ROOT/'artifacts/reviews'/cfg['parent_run']/'extracted'/cfg['parent_run']
assert identity==read(parent/'model_identity.json')
status=read(out/'run_status.json');assert status['error'] is None and all(s['exit_code']==0 for s in status['stages'])
assert status['training_enabled'] is False and status['optimizer_updates']==0 and status['elapsed_seconds']<cfg['max_wall_seconds']
print('Release source, 16 provenance files, both checkpoint bindings and tokenizer verified',flush=True)
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('reviewed_independent_auditor',out/'source/scripts/audit_v2_hard_recovery_test.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
evidence,summary,calendar=worker.audit_results(out,cfg,tok)
assert summary==read(out/'independent_test_summary.json')
assert calendar==read(out/'calendar_sql_validation.json')
audit={'passed':True,**evidence,'checkpoints_unchanged':True,'calendar_challenges_checked':True}
assert audit==read(out/'audit.json')
scratch=dest/'regenerated_counts';scratch.mkdir(exist_ok=True)
if (scratch/'recovery_counts.csv').exists():(scratch/'recovery_counts.csv').unlink()
worker.export_counts(scratch,summary);assert sha(scratch/'recovery_counts.csv')==sha(out/'recovery_counts.csv')
actions=[r for b in ('before','after') for r in worker.load(out/(b+'-recovery-actions.jsonl'))]
allturns=[r for b in ('before','after') for suffix in ('test','sample','recovery') for e in worker.load(out/(b+'-'+suffix+'-episodes.jsonl')) for r in e['turns']]
result={'zip_sha256':sha(dest/(run+'-review.zip')),'source_files_verified':len(sources),'source_matches_release_and_local':True,'provenance_files_verified':len(binding['provenance']),
 **audit,'local_audit_matches_cloud':True,'summaries_recomputed_match':True,'calendar_recomputed_matches':True,'counts_csv_matches':True,
 'native_episode_finish_reasons':dict(collections.Counter(r['generation']['finish_reason'] for r in allturns)),
 'first_action_finish_reasons':dict(collections.Counter(r['generation']['finish_reason'] for r in actions)),
 'sqlite_replay_version':sqlite3.sqlite_version,'platform_differences_resolved':2,'platform_resolution':'SQLite 3.42.0 reports near order; bundled 3.53.1 matches cloud near dot for the same two invalid SQL actions. Strict replay remains unchanged.',
 'elapsed_seconds':status['elapsed_seconds'],'candidate_adapter_sha256':cfg['candidate_adapter_sha256']}
(dest/'local_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
