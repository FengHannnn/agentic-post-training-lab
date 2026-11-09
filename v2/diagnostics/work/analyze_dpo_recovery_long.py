from pathlib import Path
import json,hashlib,collections
out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v2-20261102/extracted/dpo-recovery-train-v2-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v2-20261102');old=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102/extracted/dpo-recovery-train-v1-20261102')
read=lambda p:json.loads(p.read_text(encoding='utf8'));load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=load(old/'before-sample-episodes.jsonl');current=load(out/'before-sample-episodes.jsonl')
changes=[];all_action_text_equal=0
for a,b in zip(prior,current):
 assert (a['task_id'],a['rollout_index'],a['sample_seed'])==(b['task_id'],b['rollout_index'],b['sample_seed'])
 ta=[r['generation']['text'] for r in a['turns']];tb=[r['generation']['text'] for r in b['turns']]
 if ta==tb:all_action_text_equal+=1
 else:
  first=next((i for i,(x,y) in enumerate(zip(ta,tb)) if x!=y),min(len(ta),len(tb)))
  changes.append({'task_id':a['task_id'],'rollout_index':a['rollout_index'],'previous_before_success':a['success'],'current_before_success':b['success'],'first_differing_turn':first,
    'previous_action':a['turns'][first]['action'] if first<len(a['turns']) else None,'current_action':b['turns'][first]['action'] if first<len(b['turns']) else None})
previous_history=read(old/'training_summary.json')['history'];history=read(out/'training_summary.json')['history']
assert all({k:r[k] for k in prev}==prev for prev,r in zip(previous_history,history[:43]))
log=(out/'train.log').read_bytes()
progress={'carriage_return_count':log.count(bytes([13])),'training_bar_present':b'DPO training' in log,'final_129_of_129_present':b'129/129' in log,
 'all_epoch_completion_messages':all(('Epoch '+str(i)+'/3 completed').encode() in log for i in (1,2,3)),'training_steps_jsonl_rows':len(load(out/'training_steps.jsonl'))}
results={'first_epoch_adapter_matches_previous':read(out/'checkpoint-0043/complete.json')['adapter_sha256']['adapter_model.safetensors']==read(old/'final/complete.json')['adapter_sha256']['adapter_model.safetensors'],
 'first_43_update_metrics_identical':True,'reference_cache_bytes_identical':sha(out/'reference_scores.json')==sha(old/'reference_scores.json'),
 'cross_run_initial_model_sample_trajectories_with_same_actions':all_action_text_equal,'cross_run_initial_model_sample_changes':changes,
 'within_current_run_paired_sample':read(out/'evaluation_summary.json')['paired_development_sample'],'terminal_progress_evidence':progress}
(dest/'cross_run_and_progress_analysis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(results,ensure_ascii=False,indent=2))
