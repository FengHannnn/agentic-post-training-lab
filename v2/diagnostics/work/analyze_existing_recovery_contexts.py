"""Attribute previously audited development actions without model inference."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.tasks import make_tasks
from apt_v2.recovery_diagnostics import diagnose_action,diagnostic_summary
from apt_v2.dpo_recovery_training import validate_generation
from transformers import AutoTokenizer
from common import file_hash
out=ROOT/'artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102'
dest=ROOT/'artifacts/diagnostics/recovery-context-diagnostics-v1-20261102';dest.mkdir(parents=True,exist_ok=True)
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
contexts={c['context_id']:c for c in load(out/'recovery_contexts.jsonl')};tasks={t.id:t for t in make_tasks()}
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
rows=[];sources={}
for old,branch in [('before','before'),('dpo','dpo'),('after','sft')]:
    path=out/(old+'-recovery-actions.jsonl');sources[path.name]=file_hash(path)
    for row in load(path):
        c=contexts[row['context_id']];g=row['generation'];validate_generation(tok,c['messages'],g,'greedy' if row['mode']=='greedy' else 'dpo_sample')
        actual,diagnostic=diagnose_action(tasks[c['task_id']],c,g['text'])
        assert actual==row['result']['row']
        assert diagnostic['productive']==row['result']['productive']
        rows.append({**{k:v for k,v in row.items() if k not in ('result','model_branch')},'model_branch':branch,'row':actual,'diagnostic':diagnostic})
with (dest/'existing_development_actions.jsonl').open('w',encoding='utf8') as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
summary=diagnostic_summary(rows,('before','dpo','sft'))
(dest/'existing_development_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
audit={'passed':True,'source_run':'chosen-only-sft-v1-20261102','actions_replayed':len(rows),'source_files_sha256':sources,'new_model_generations':0}
(dest/'existing_development_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
for b in summary:
    print(b,json.dumps(summary[b]['development']['sample']['overall'],ensure_ascii=False))
