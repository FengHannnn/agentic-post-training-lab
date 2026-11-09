from pathlib import Path
import json,hashlib,sys,importlib.util,math,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102/extracted/dpo-recovery-train-v1-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/dpo-recovery-train-v1.json')
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
runs={run:ROOT/'artifacts/reviews'/run/'extracted'/run for run in (cfg['initial_run'],cfg['data_run'])}
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts;assert sha(runs[parts[1]].joinpath(*parts[2:]))==h,rel
assert sha(runs[cfg['initial_run']]/'final/adapter/adapter_model.safetensors')==cfg['adapter_sha256']
assert sha(out/'pairs.jsonl')==cfg['pairs_sha256']==sha(runs[cfg['data_run']]/'pairs.jsonl')
identity=read(out/'model_identity.json');assert identity==read(runs[cfg['initial_run']]/'model_identity.json')
for n,h in identity['files_sha256'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
final=read(out/'final/complete.json')
for n,h in final['adapter_sha256'].items():assert sha(out/'final/adapter'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('reviewed_recovery_auditor',out/'source/scripts/audit_v2_dpo_recovery.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from transformers import AutoTokenizer
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.dpo_recovery_training import validate_training_pairs,verify_reference
pairs=worker.load(out/'pairs.jsonl');validate_training_pairs(pairs)
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True)
encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
verify_reference(pairs,encoded,read(out/'reference_scores.json'),cfg['adapter_sha256'])
pre=read(out/'before_pair_scores.json');post=read(out/'after_pair_scores.json')
for scores in (pre,post):
 assert scores['pairs']==169 and len(scores['metrics'])==169
 for k,v in scores['means'].items():assert abs(sum(m[k] for m in scores['metrics'])/169-v)<1e-8
 for m in scores['metrics']:assert all(math.isfinite(v) for v in m.values()) and abs(m['loss']-math.log1p(math.exp(-cfg['beta']*m['relative_margin'])))<1e-5
assert abs(pre['means']['loss']-math.log(2))<1e-5
train=read(out/'training_summary.json')
assert final['steps']==43 and final['cursor']==169 and final['source_dpo_updates']==11 and final['total_dpo_updates']==54
assert train['optimizer_start']=='fresh_adam' and train['optimizer_updates']==43 and train['pair_exposures']==169 and train['frozen_parameter_gradients']==0
assert len(train['history'])==43 and [r['pairs_seen'] for r in train['history']]==list(range(4,169,4))+[169]
evidence=worker.audit_evaluations(out,cfg,tok)
expected={'passed':True,**{k:v for k,v in evidence.items() if k not in ('paired_greedy','paired_development_sample')},'preference_pairs':169,'reference_inputs_verified':169,'optimizer_updates':43,'reference_and_initial_checkpoint_unchanged':True,'test_split_used':False}
assert expected==read(out/'audit.json')
assert {**evidence,'status':'awaiting_local_review','optimizer_updates':43,'recovery_comparison':read(out/'recovery_comparison.json')}==read(out/'evaluation_summary.json')
(dest/'local_audit.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2),encoding='utf8')
from safetensors.torch import load_file
import torch
before=load_file(str(runs[cfg['initial_run']]/'final/adapter/adapter_model.safetensors'));after=load_file(str(out/'final/adapter/adapter_model.safetensors'))
assert before.keys()==after.keys() and all(torch.isfinite(t).all() for t in after.values())
groups={}
for key in ('rejected_source','chosen_tool'):
 values=[p['rejected_source'] if key=='rejected_source' else json.loads(p['chosen'])['tool'] for p in pairs];groups[key]={}
 for value in sorted(set(values)):
  ms=[m for v,m in zip(values,post['metrics']) if v==value]
  groups[key][value]={'pairs':len(ms),'mean_loss':sum(m['loss'] for m in ms)/len(ms),'mean_relative_margin':sum(m['relative_margin'] for m in ms)/len(ms),'positive_margin':sum(m['relative_margin']>0 for m in ms)}
result={'zip_sha256':sha(dest/'dpo-recovery-train-v1-20261102-review.zip'),'source_files_verified':len(sources),'source_matches_local':True,
 'provenance_files_verified':len(binding['provenance']),'tokenizer_files_verified':sum((out/'tokenizer'/n).exists() for n in identity['files_sha256']),
 'episodes_replayed':evidence['episodes_replayed'],'episode_turns_replayed':evidence['episode_turns_replayed'],'recovery_actions_replayed':evidence['recovery_actions_replayed'],
 'reference_inputs_verified':169,'local_audit_matches_cloud':True,'adapter_sha256':sha(out/'final/adapter/adapter_model.safetensors'),
 'adapter_tensors_changed':sum(not torch.equal(before[k],after[k]) for k in before),'before_pair_means':pre['means'],'after_pair_means':post['means'],
 'positive_relative_margins':sum(m['relative_margin']>0 for m in post['metrics']),'pair_groups':groups,
 'optimizer_state_review':'cloud audit verifies state hash and per-parameter step43; state.pt retained on cloud'}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
