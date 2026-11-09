from pathlib import Path
import json,sys,copy
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path.insert(0,str(ROOT/'src'))
from transformers import AutoTokenizer
from apt_v2.dpo import verified_pairs
from apt_v2.tasks import make_tasks
r=ROOT/'artifacts/reviews/sft-diagnostics-v1-20261101/offline-review/recovered'
tok=AutoTokenizer.from_pretrained(r/'tokenizer',local_files_only=True)
tasks={t.id:t for t in make_tasks() if t.split=='train'}
episodes=[json.loads(l) for l in (r/'train-sft-episodes.jsonl').read_text(encoding='utf8').splitlines()]
# Historical 63-step failures exercise native token encoding only; never use these as new 126-step rollout data.
counts=[]
for original in episodes[:3]:
    e=copy.deepcopy(original);e['rollout_id']='historical-63step-native-token-fixture-'+e['task_id']
    for row in e['turns']:row['generation']['mode']='dpo_sample'
    pairs=verified_pairs(tasks[e['task_id']],e,tok,2048,2)
    counts.append({'task_id':e['task_id'],'pairs':len(pairs),'tokens':[(p['chosen_tokens'],p['rejected_tokens']) for p in pairs]})
assert all(c['pairs']>0 for c in counts)
print(json.dumps({'fixture':'historical 63-step train failures; schema-only mode adaptation, no new rollout claim','native_token_smoke':counts}))
