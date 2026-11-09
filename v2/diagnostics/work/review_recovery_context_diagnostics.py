from pathlib import Path
import json,hashlib,sys,importlib.util,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');run='recovery-context-diagnostics-v1-20261102'
dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/recovery-context-diagnostics-v1.json')
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
assert sha(out/'config.json')==binding['config_sha256']
assert sha(out/'pairs.jsonl')==binding['pairs_sha256']==cfg['pairs_sha256']
assert sha(out/'contexts.jsonl')==binding['contexts_sha256']
assert sha(out/'context_membership.jsonl')==binding['membership_sha256']
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts
 local=ROOT/'artifacts/reviews'/parts[1]/'extracted'/parts[1]
 assert sha(local.joinpath(*parts[2:]))==h,rel
identity=read(out/'model_identity.json')
for branch,spec in cfg['checkpoints'].items():
 prior=ROOT/'artifacts/reviews'/spec['run']/'extracted'/spec['run']
 assert identity==read(prior/'model_identity.json')
 assert sha(prior/'final/adapter/adapter_model.safetensors')==spec['adapter_sha256']
for n,h in identity['files_sha256'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('frozen_context_auditor',out/'source/scripts/diagnose_v2_recovery_contexts.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
print('Source, three adapters, provenance and tokenizer verified',flush=True)
audit=worker.audit_diagnostics(out,cfg,tok);assert audit==read(out/'audit.json')
summary=read(out/'diagnostic_summary.json')
assert worker.comparison_rates(summary)==read(out/'train_development_comparison.json')
scratch=dest/'regenerated_counts';scratch.mkdir(exist_ok=True)
(scratch/'diagnostic_summary.json').write_text(json.dumps(summary),encoding='utf8')
if (scratch/'diagnostic_counts.csv').exists():(scratch/'diagnostic_counts.csv').unlink()
worker.export_csv(scratch)
assert sha(scratch/'diagnostic_counts.csv')==sha(out/'diagnostic_counts.csv')
actions={b:worker.load(out/(b+'-actions.jsonl')) for b in cfg['checkpoints']}
finish={b:dict(collections.Counter(r['generation']['finish_reason'] for r in rows)) for b,rows in actions.items()}
log=(out/'evaluate.log').read_bytes();status=read(out/'run_status.json')
assert status['error'] is None and all(s['exit_code']==0 for s in status['stages']) and status['training_updates']==0
assert len(actions)==3 and sum(map(len,actions.values()))==1344
result={'zip_sha256':sha(dest/(run+'-review.zip')),'source_files_verified':len(sources),'source_matches_local':True,
 'provenance_files_verified':len(binding['provenance']),**audit,'local_audit_matches_cloud':True,
 'csv_recomputed_matches':True,'comparison_rates_recomputed_matches':True,'finish_reason_counts':finish,
 'progress_1344_present':b'1344/1344' in log,'elapsed_seconds':status['elapsed_seconds'],
 'adapter_hashes':{b:s['adapter_sha256'] for b,s in cfg['checkpoints'].items()}}
(dest/'local_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
for b in summary:
 for split in ('train','development'):
  s=summary[b][split]['sample']['overall'];e=s['evidence'];q=s['sql']
  print(b,split,'sample',json.dumps({'actions':s['actions'],'productive':s['productive'],'tool_match':s['tool_matches_expert'],
  'evidence_correct_on_expected_tool':[e['expected_tool_all_references_verified'],e['expected_tool_actions']],
  'query_selected_required':[q['query_chosen_when_required'],q['query_required_actions']],
  'target_verified':q['target_verified_when_required'],'sql_attempts':q['attempts'],'sql_status':q['status_counts']}))
