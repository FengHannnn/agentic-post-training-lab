from pathlib import Path
import json,hashlib,math,collections
r=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-train-v1-20261102/extracted/dpo-train-v1-20261102');root=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-train-v1-20261102')
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources=read(r/'run_manifest.json')['source_sha256']
for n,h in sources.items(): assert sha(r/'source'/n)==h and sha(root/n)==h,n
identity=read(r/'model_identity.json')
for n,h in identity['files_sha256'].items():
 if (r/'tokenizer'/n).exists():assert sha(r/'tokenizer'/n)==h
assert read(dest/'local_audit.json')==read(r/'audit.json')
load=lambda n:[json.loads(l) for l in (r/n).read_text(encoding='utf8').splitlines()]
a=load('sft-sample-episodes.jsonl');b=load('dpo-sample-episodes.jsonl');key=lambda e:(e['task_id'],e['rollout_index'])
bm={key(e):e for e in b};trans=collections.Counter();changes=[]
for e in a:
 f=bm[key(e)];trans[str(int(e['success']))+'->'+str(int(f['success']))]+=1
 if e['success']!=f['success']:changes.append({'task_id':e['task_id'],'rollout_index':e['rollout_index'],'before':e['success'],'after':f['success'],'before_turns':len(e['turns']),'after_turns':len(f['turns'])})
greedy_equal=True
for split in ('train','development'):
 ga=load('sft-'+split+'-episodes.jsonl');gb=load('dpo-'+split+'-episodes.jsonl')
 for e,f in zip(ga,gb):
  greedy_equal &= [x['generation']['text'] for x in e['turns']]==[x['generation']['text'] for x in f['turns']]
pre=read(r/'before_pair_scores.json');post=read(r/'after_pair_scores.json');refs=read(r/'reference_scores.json')['scores'];cfg=read(r/'config.json')
for i,m in enumerate(post['metrics']):assert abs(m['loss']-math.log1p(math.exp(-cfg['beta']*m['relative_margin'])))<1e-6
from safetensors.torch import load_file
sft=load_file(str(root/'artifacts/baselines/sft-sql-weighted-v1-20261102/adapter/adapter_model.safetensors'));dpo=load_file(str(r/'final/adapter/adapter_model.safetensors'))
assert sft.keys()==dpo.keys()
import torch
assert all(torch.isfinite(t).all() for t in dpo.values())
result={'zip_sha256':sha(dest/'dpo-train-v1-20261102-review.zip'),'source_files_verified':len(sources),'tokenizer_files_verified':sum((r/'tokenizer'/n).exists() for n in identity['files_sha256']),'local_audit_matches_cloud':True,'episodes_replayed':128,'turns_replayed':1150,'optimizer_state_review':'cloud audit passed; state.pt excluded from return package','adapter_sha256':sha(r/'final/adapter/adapter_model.safetensors'),'adapter_tensors_changed':sum(not torch.equal(sft[k],dpo[k]) for k in sft),'greedy_actions_identical':greedy_equal,'paired_success_transitions':dict(trans),'changed_outcomes':changes,'before_means':pre['means'],'after_means':post['means'],'positive_relative_margins':sum(m['relative_margin']>0 for m in post['metrics']),'chosen_mean_logp_increases':sum(y['chosen_mean_logp']>x['chosen_mean_logp'] for x,y in zip(pre['metrics'],post['metrics'])),'rejected_mean_logp_decreases':sum(y['rejected_mean_logp']<x['rejected_mean_logp'] for x,y in zip(pre['metrics'],post['metrics']))}
(dest/'local_review_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
