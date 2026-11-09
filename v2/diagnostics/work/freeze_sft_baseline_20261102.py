from pathlib import Path
import json,hashlib,shutil
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2').resolve();base=ROOT/'artifacts/baselines';base.mkdir(parents=True,exist_ok=True)
baseline_id='sft-sql-weighted-v1-20261102';target=(base/baseline_id).resolve();staging=(base/(baseline_id+'.staging')).resolve()
source=(ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101').resolve()
test=(ROOT/'artifacts/reviews/sft-independent-test-v1-20261102/extracted/sft-independent-test-v1-20261102').resolve()
assert target.parent==base.resolve() and staging.parent==base.resolve() and source.is_relative_to(ROOT) and test.is_relative_to(ROOT)
if target.exists() or staging.exists():raise FileExistsError('fresh baseline directory required')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=read(test/'independent_test_summary.json')
assert summary['acceptance']['eligible_for_local_review']
local_audit=ROOT/'artifacts/reviews/sft-independent-test-v1-20261102/local_review_summary.json'
assert read(local_audit)['local_audit_matches_cloud']
complete=read(source/'final/complete.json')
for n,h in complete['adapter_sha256'].items():assert sha(source/'final/adapter'/n)==h
assert sha(source/'final/state.pt')==complete['state_sha256']
assert sha(source/'final/adapter/adapter_model.safetensors')==summary['candidate_adapter_sha256']
shutil.copytree(source/'final',staging)
for name in ('model_identity.json','training_summary.json','data_summary.json'):
    shutil.copyfile(source/name,staging/name)
shutil.copyfile(source/'config.json',staging/'training_config.json')
shutil.copytree(test/'tokenizer',staging/'tokenizer')
shutil.copyfile(test/'config.json',staging/'test_config.json')
shutil.copyfile(test/'independent_test_summary.json',staging/'test_summary.json')
shutil.copyfile(local_audit,staging/'test_local_audit.json')
metrics=read(source/'comparison_summary.json')['weighted']
metadata={'baseline_id':baseline_id,'status':'frozen','frozen_date':'2026-11-02','timezone':'Asia/Shanghai',
 'model_id':'Qwen/Qwen2.5-1.5B-Instruct','revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
 'adapter_sha256':summary['candidate_adapter_sha256'],'total_optimizer_updates':126,
 'roles':['common_sft_parent_for_dpo','common_sft_parent_for_grpo','fixed_sft_reference_for_rl'],
 'optimizer_state_preserved_for':'reproducibility and explicit future continuation choices',
 'evaluation':{'train':metrics['train'],'development':metrics['development'],'test':summary['models']['weighted']},
 'test_exposure':'first model evaluation completed; retain result and exposure record',
 'source_checkpoint':source.relative_to(ROOT).as_posix()+'/final',
 'test_review':'artifacts/reviews/sft-independent-test-v1-20261102',
 'future_branch_outputs':'separate new run directories; verify baseline manifest before use'}
(staging/'freeze_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
files={p.relative_to(staging).as_posix():sha(p) for p in sorted(staging.rglob('*')) if p.is_file()}
(staging/'baseline_manifest.json').write_text(json.dumps({'baseline_id':baseline_id,'files_sha256':files},indent=2)+'\n',encoding='utf8')
for n,h in files.items():assert sha(staging/n)==h
# Both resolved paths were checked to stay directly inside artifacts/baselines above.
staging.rename(target)
receipt={'baseline_id':baseline_id,'path':target.relative_to(ROOT).as_posix(),'files_verified':len(files),
 'adapter_sha256':sha(target/'adapter/adapter_model.safetensors'),'manifest_sha256':sha(target/'baseline_manifest.json'),
 'test_successes':4,'test_tasks':4,'recovered':2,'fault_tasks':2,'state':'frozen'}
(base/(baseline_id+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
candidate_path=ROOT/'artifacts/candidates/sql-sft-weighted-v1-20261101/candidate_identity.json'
shutil.copyfile(candidate_path,ROOT/'artifacts/reviews/sft-independent-test-v1-20261102/candidate_identity_before_freeze.json')
candidate=read(candidate_path);candidate.update(status='promoted_to_frozen_sft_baseline',independent_test_run=True,
 test_successes=4,test_tasks=4,baseline_id=baseline_id,baseline_path=receipt['path'])
candidate_path.write_text(json.dumps(candidate,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
