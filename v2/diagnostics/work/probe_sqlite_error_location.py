import sqlite3,json
from pathlib import Path
p=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/hard-recovery-independent-test-v1-20261103/extracted/hard-recovery-independent-test-v1-20261103/before-sample-episodes.jsonl')
e=next(e for e in map(json.loads,p.read_text(encoding='utf8').splitlines()) if e['task_id']=='test-001' and e['rollout_index']==1)
sql=json.loads(e['turns'][10]['generation']['text'])['arguments']['sql']
print('sqlite',sqlite3.sqlite_version)
for q in ['SELECT order.gross_cents FROM orders order',sql]:
 try:sqlite3.connect(':memory:').execute(q)
 except sqlite3.Error as exc:print(str(exc))
