from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from apt_v2.tasks import make_tasks
from apt_v2.dpo_recovery import build_context,assess
out=ROOT/'artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102'
row=next(json.loads(l) for l in (out/'after-recovery-actions.jsonl').read_text(encoding='utf8').splitlines() if json.loads(l)['context_id']=='development-000:sql_unverified_requery' and json.loads(l)['mode']=='greedy')
task=next(t for t in make_tasks() if t.id==row['task_id'])
c=build_context(task,row['scenario'],allowed_split='development')
actual,reason=assess(task,c,row['generation']['text'],allowed_split='development')
print(json.dumps({'expected_tool':json.loads(c['chosen'])['tool'],'actual_tool':actual['action'].get('tool'),'reason':reason,'tool_error':actual['tool_error']},ensure_ascii=False))
assert actual['action'].get('tool')==json.loads(c['chosen'])['tool'],'unverified monthly evidence requires returning to query_sql'
