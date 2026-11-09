from pathlib import Path
import json,hashlib,sys,importlib.util,math
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');run='chosen-only-sft-v1-20261102';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/chosen-only-sft-v1.json')
assert binding['source_sha256']==sources
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
assert sha(out/'config.json')==binding['config_sha256']
runs={r:ROOT/'artifacts/reviews'/r/'extracted'/r for r in (cfg['initial_run'],cfg['data_run'],cfg['comparison_run'])}
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts;assert sha(runs[parts[1]].joinpath(*parts[2:]))==h,rel
assert sha(runs[cfg['initial_run']]/'final/adapter/adapter_model.safetensors')==cfg['adapter_sha256']==binding['baseline_adapter_sha256']
assert sha(runs[cfg['comparison_run']]/'final/adapter/adapter_model.safetensors')==cfg['comparison_adapter_sha256']
assert sha(out/'pairs.jsonl')==cfg['pairs_sha256']==binding['pairs_sha256']==sha(runs[cfg['data_run']]/'pairs.jsonl')
identity=read(out/'model_identity.json');assert identity==read(runs[cfg['initial_run']]/'model_identity.json')
for n,h in identity['files_sha256'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
final=read(out/'final/complete.json')
for n,h in final['adapter_sha256'].items():assert sha(out/'final/adapter'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
from transformers import AutoTokenizer
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256,verify_sft_updates
from apt_v2.dpo_recovery import encode_recovery_pair
spec=importlib.util.spec_from_file_location('reviewed_recovery_auditor',out/'source/scripts/audit_v2_dpo_recovery.py');worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
pairs=worker.load(out/'pairs.jsonl');examples=chosen_examples(pairs)
assert examples==worker.load(out/'examples.jsonl') and data_sha256(examples)==binding['examples_sha256']
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
encoded=encode_chosen(tok,examples,cfg['max_length']);assert data_sha256(encoded)==binding['encoded_sha256']
for pair,s in zip(pairs,encoded):
 n=s['prompt_length'];assert s['labels'][:n]==[-100]*n and s['labels'][n:]==s['input_ids'][n:]
 d=encode_recovery_pair(tok,pair,cfg['max_length']);assert d['prompt']==s['input_ids'][:n] and d['chosen']==s['input_ids'][n:]
pre=read(out/'before_chosen_scores.json');post=read(out/'after_chosen_scores.json')
for scores in (pre,post):
 assert scores['examples']==169 and len(scores['metrics'])==169
 for s,m in zip(encoded,scores['metrics']):
  probs=m['token_log_probs'];assert m['input_id']==data_sha256(s) and len(probs)==s['response_length']
  assert all(math.isfinite(v) and v<=1e-6 for v in probs)
  assert abs(-sum(probs)/len(probs)-m['loss'])<1e-6
 assert abs(sum(m['loss'] for m in scores['metrics'])/169-scores['mean_loss'])<1e-8
ref=read(runs[cfg['comparison_run']]/'reference_scores.json')
assert ref['baseline_adapter_sha256']==cfg['adapter_sha256'] and ref['pair_ids']==[p['pair_id'] for p in pairs]
maxdiff=max(abs(a-b) for m,s in zip(pre['metrics'],ref['scores']) for a,b in zip(m['token_log_probs'],s['chosen']))
assert maxdiff<=1e-5
train=read(out/'training_summary.json');verify_sft_updates(cfg,final,train)
assert worker.load(out/'training_steps.jsonl')==train['history']
assert train['rejected_used_in_loss'] is False and train['reference_used_in_loss'] is False
assert train['supervised_tokens_per_epoch']==sum(s['response_length'] for s in encoded)==17581
for step,epoch in ((43,1),(86,2)):
 c=read(out/f'checkpoint-{step:04}/complete.json');assert c['steps']==step and c['cursor']==169*epoch and c['epochs_completed']==epoch
print('Input hashes, 169 chosen examples, encoding, initial scores and 129 updates verified',flush=True)
evidence=worker.audit_evaluations(out,cfg,tok)
expected={'passed':True,**{k:v for k,v in evidence.items() if not k.startswith('paired_')},'chosen_examples':169,'supervised_tokens_per_epoch':17581,'optimizer_updates':129,'chosen_only_supervision':True,'initial_chosen_scores_match_dpo':True,'initial_and_comparison_checkpoints_unchanged':True,'test_split_used':False}
assert expected==read(out/'audit.json')
assert {**evidence,'status':'awaiting_local_review','algorithm':'chosen_only_sft','optimizer_updates':129,'recovery_comparison':read(out/'recovery_comparison.json')}==read(out/'evaluation_summary.json')
(dest/'local_audit.json').write_text(json.dumps(expected,ensure_ascii=False,indent=2),encoding='utf8')
from safetensors.torch import load_file
import torch
before=load_file(str(runs[cfg['initial_run']]/'final/adapter/adapter_model.safetensors'));after=load_file(str(out/'final/adapter/adapter_model.safetensors'))
assert before.keys()==after.keys() and all(torch.isfinite(t).all() for t in after.values())
log=(out/'train.log').read_bytes()
result={'zip_sha256':sha(dest/(run+'-review.zip')),'source_files_verified':len(sources),'source_matches_local':True,'provenance_files_verified':len(binding['provenance']),**expected,'local_audit_matches_cloud':True,'adapter_sha256':sha(out/'final/adapter/adapter_model.safetensors'),'adapter_tensors_changed':sum(not torch.equal(before[k],after[k]) for k in before),'initial_chosen_score_max_abs_diff':maxdiff,'before_chosen_mean_loss':pre['mean_loss'],'after_chosen_mean_loss':post['mean_loss'],'unique_context_chosen_texts':len({json.dumps([e['messages'],e['response']],sort_keys=True) for e in examples}),'epochs':3,'pair_exposures':507,'training_progress_129_present':b'129/129' in log,'progress_carriage_returns':log.count(b'\r'),'optimizer_state_review':'Cloud audit verifies state hash and per-parameter step129; state.pt retained on cloud; local review verifies metadata and streamed history.'}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
