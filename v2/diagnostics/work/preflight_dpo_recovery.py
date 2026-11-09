from pathlib import Path
import json,sys,hashlib,collections,shutil
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path.insert(0,str(ROOT/'src'))
from transformers import AutoTokenizer
from apt_v2.tasks import make_tasks
from apt_v2.dpo_recovery import SCENARIOS,build_context,make_pair,select_pairs,encode_recovery_pair
out=ROOT/'artifacts/datasets/dpo-recovery-scripted-v1-20261102';out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/reviews/dpo-train-v1-20261102/extracted/dpo-train-v1-20261102/tokenizer',local_files_only=True)
anchor_path=ROOT/'artifacts/reviews/dpo-collect-v1-20261102/extracted/dpo-collect-v1-20261102/pairs.jsonl'
anchors=[json.loads(l) for l in anchor_path.read_text(encoding='utf8').splitlines()]
tasks=[t for t in make_tasks() if t.split=='train'];tm={t.id:t for t in tasks}
contexts=[build_context(t,s) for t in tasks for s in SCENARIOS]
scripted=[make_pair(tm[c['task_id']],c,tok,2048) for c in contexts]
assert all(p is not None for p in scripted)
pairs=select_pairs(anchors,scripted,[]);encoded=[encode_recovery_pair(tok,p,2048) for p in pairs]
for name,rows in [('contexts',contexts),('scripted_pairs',scripted),('pairs',pairs)]:
 (out/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf8')
shutil.copyfile(anchor_path,out/'anchor_pairs.jsonl')
result={'source':'local deterministic preflight with pinned real tokenizer','anchors':len(anchors),'scripted_recovery_pairs':len(scripted),'pairs':len(pairs),
 'train_tasks':len(tasks),'contexts':len(contexts),'pairs_by_chosen_tool':dict(collections.Counter(json.loads(p['chosen'])['tool'] for p in pairs)),
 'recovery_pairs_by_scenario':dict(collections.Counter(p['scenario'] for p in scripted)),
 'max_pair_tokens':max(len(e['prompt'])+max(len(e['chosen']),len(e['rejected'])) for e in encoded),
 'pairs_sha256':hashlib.sha256((out/'pairs.jsonl').read_bytes()).hexdigest(),'native_samples_collected':0,'optimizer_updates':0}
(out/'preflight_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,ensure_ascii=False,indent=2))
