from pathlib import Path
import json,hashlib,sys,importlib.util,math,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='hard-recovery-sft-weighted-v1-20261103';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(out/'config.json');binding=read(out/'input_binding.json');sources=read(out/'run_manifest.json')['source_sha256']
assert cfg==read(ROOT/'configs/hard-recovery-sft-weighted-v1.json')
assert binding['source_sha256']==sources
for n,h in sources.items():assert sha(out/'source'/n)==h and sha(ROOT/n)==h,n
assert sha(out/'config.json')==binding['config_sha256']
for n,k in [('pairs.jsonl','pairs_sha256'),('weights.json','weights_sha256'),('contexts.jsonl','contexts_sha256'),('context_membership.jsonl','membership_sha256')]:assert sha(out/n)==binding[k]
assert sha(out/'pairs.jsonl')==cfg['pairs_sha256']
for rel,h in binding['provenance'].items():
 parts=Path(rel).parts;prior=ROOT/'artifacts/reviews'/parts[1]/'extracted'/parts[1]
 assert sha(prior.joinpath(*parts[2:]))==h,rel
parent=ROOT/'artifacts/reviews'/cfg['initial_run']/'extracted'/cfg['initial_run']
assert sha(parent/'final/adapter/adapter_model.safetensors')==cfg['adapter_sha256']
identity=read(out/'model_identity.json');assert identity==read(parent/'model_identity.json')
for n,h in identity['files_sha256'].items():
 if (out/'tokenizer'/n).exists():assert sha(out/'tokenizer'/n)==h,n
final=read(out/'final/complete.json')
for n,h in final['adapter_sha256'].items():assert sha(out/'final/adapter'/n)==h,n
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
spec=importlib.util.spec_from_file_location('reviewed_hard_auditor',out/'source/scripts/audit_v2_hard_recovery_sft.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
from transformers import AutoTokenizer
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.hard_recovery_sft import recovery_weights,verify_weighted_updates
pairs=worker.load(out/'pairs.jsonl');examples=chosen_examples(pairs);weights,ws=recovery_weights(pairs,cfg['hard_weight'])
assert examples==worker.load(out/'examples.jsonl') and data_sha256(examples)==binding['examples_sha256']
assert read(out/'weights.json')=={'weights':weights,'summary':ws}
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
encoded=encode_chosen(tok,examples,cfg['max_length']);assert data_sha256(encoded)==binding['encoded_sha256']
for s in encoded:
 n=s['prompt_length'];assert s['labels'][:n]==[-100]*n and s['labels'][n:]==s['input_ids'][n:] and s['input_ids'][-1]==tok.eos_token_id
scores={}
for name in ('before_chosen_scores.json','after_chosen_scores.json'):
 data=read(out/name);assert data['examples']==169 and len(data['metrics'])==169
 for s,m in zip(encoded,data['metrics']):
  ps=m['token_log_probs'];assert m['input_id']==data_sha256(s) and len(ps)==s['response_length'] and all(math.isfinite(v) and v<=1e-6 for v in ps)
  assert math.isfinite(m['loss']) and abs(-sum(ps)/len(ps)-m['loss'])<1e-6
 assert abs(sum(m['loss'] for m in data['metrics'])/169-data['mean_loss'])<1e-8
 assert abs(sum(m['loss']*w for m,w in zip(data['metrics'],weights))/169-data['mean_weighted_loss'])<1e-8
 scores[name]=data
cached=read(parent/'after_chosen_scores.json');before=scores['before_chosen_scores.json']
for a,b in zip(before['metrics'],cached['metrics']):assert a['input_id']==b['input_id'] and len(a['token_log_probs'])==len(b['token_log_probs'])
maxdiff=max(abs(x-y) for a,b in zip(before['metrics'],cached['metrics']) for x,y in zip(a['token_log_probs'],b['token_log_probs']))
assert maxdiff<=1e-5
train=read(out/'training_summary.json');verify_weighted_updates(cfg,final,train)
assert train['history']==worker.load(out/'training_steps.jsonl') and train['weight_summary']==ws
assert train['rejected_used_in_loss'] is False and train['reference_used_in_loss'] is False and train['token_weighting'] is False
assert train['supervised_tokens_per_epoch']==sum(s['response_length'] for s in encoded)==17581
for step in cfg['checkpoint_steps']:
 p=out/f'checkpoint-{step:04}';c=read(p/'complete.json')
 assert c['algorithm']=='hard_recovery_weighted_sft' and c['steps']==step and c['epochs_completed']==step//43 and c['cursor']==169*(step//43)
 assert c['total_sft_updates']==255+step and c['total_dpo_updates']==11 and read(p/'binding.json')==binding
assert read(out/'final/binding.json')==binding
print('Sources, provenance, data, normalized weights and 258-update history verified',flush=True)
tasks=worker.audit_task_evaluations(out,cfg,tok)
print('64 complete episodes replayed',flush=True)
recovery=worker.audit_diagnostics(out,{**cfg,'checkpoints':{b:{} for b in cfg['evaluation_branches']}},tok)
print('896 initial/final recovery actions replayed',flush=True)
checkpoint_actions=worker.audit_checkpoint_actions(out,cfg,tok)
print('420 intermediate checkpoint actions replayed',flush=True)
summary=read(out/'diagnostic_summary.json');mid=read(out/'checkpoint_diagnostics.json')
progress=[]
for name,step in [('before',0),*[(f'checkpoint-{s:04}',s) for s in cfg['checkpoint_steps']],('after',cfg['max_steps'])]:
 values=summary[name]['train']['greedy'] if name in ('before','after') else mid[name]
 complete=read(parent/'final/complete.json') if name=='before' else read(out/('final' if name=='after' else name)/'complete.json')
 progress.append({'checkpoint':name,'new_sft_updates':step,'total_sft_updates':cfg['source_sft_updates']+step,
  'adapter_sha256':complete['adapter_sha256']['adapter_model.safetensors'],'train_greedy':{k:v for k,v in values.items() if k!='actions_sha256'}})
assert {'selection':'fixed_final','checkpoints':progress}==read(out/'checkpoint_progress.json')
assert worker.comparison_rates(summary)==read(out/'train_development_comparison.json')
scratch=dest/'regenerated_counts';scratch.mkdir(exist_ok=True)
(scratch/'diagnostic_summary.json').write_text(json.dumps(summary),encoding='utf8')
if (scratch/'diagnostic_counts.csv').exists():(scratch/'diagnostic_counts.csv').unlink()
worker.export_csv(scratch);assert sha(scratch/'diagnostic_counts.csv')==sha(out/'diagnostic_counts.csv')
evidence={**tasks,'final_recovery_actions_replayed':recovery['actions_replayed'],'checkpoint_recovery_actions_replayed':checkpoint_actions,
 'train_contexts':recovery['train_contexts'],'development_contexts':recovery['development_contexts'],
 'training_contexts_exactly_in_pairs':recovery['training_contexts_exactly_in_pairs'],'development_contexts_in_training_pairs':0,
 'chosen_examples':169,'optimizer_updates':258,'hard_examples':84,'checkpoint_selection':'fixed_final'}
audit={'passed':True,**{k:v for k,v in evidence.items() if not k.startswith('paired_')},'weights_verified':True,'chosen_only_supervision':True,
 'parent_initial_probabilities_match':True,'parent_checkpoint_unchanged':True,'test_split_used':False}
assert audit==read(out/'audit.json')
assert {**evidence,'status':'awaiting_local_review'}==read(out/'evaluation_summary.json')
from safetensors.torch import load_file
import torch
a=load_file(str(parent/'final/adapter/adapter_model.safetensors'));b=load_file(str(out/'final/adapter/adapter_model.safetensors'))
assert a.keys()==b.keys() and all(torch.isfinite(t).all() for t in b.values())
status=read(out/'run_status.json');assert status['error'] is None and all(s['exit_code']==0 for s in status['stages'])
all_actions=[r for branch in ('before','after',*[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']]) for r in worker.load(out/(branch+'-actions.jsonl'))]
finish=dict(collections.Counter(r['generation']['finish_reason'] for r in all_actions))
result={'zip_sha256':sha(dest/(run+'-review.zip')),'source_files_verified':len(sources),'source_matches_local':True,
 'provenance_files_verified':len(binding['provenance']),**audit,'local_audit_matches_cloud':True,'counts_csv_matches':True,
 'checkpoint_progress_recomputed_matches':True,'train_development_comparison_recomputed_matches':True,
 'adapter_sha256':sha(out/'final/adapter/adapter_model.safetensors'),'adapter_tensors_changed':sum(not torch.equal(a[k],b[k]) for k in a),
 'initial_parent_logp_max_abs_diff':maxdiff,'before_chosen_mean_loss':before['mean_loss'],
 'after_chosen_mean_loss':scores['after_chosen_scores.json']['mean_loss'],'before_weighted_loss':before['mean_weighted_loss'],
 'after_weighted_loss':scores['after_chosen_scores.json']['mean_weighted_loss'],'recovery_finish_reasons':finish,
 'training_progress_258_present':b'258/258' in (out/'train.log').read_bytes(),
 'optimizer_state_review':'Cloud audit verifies state hashes and final per-parameter Adam step258; intermediate adapters and state.pt retained on cloud; local review verifies returned metadata/history and final adapter.'}
(dest/'local_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
