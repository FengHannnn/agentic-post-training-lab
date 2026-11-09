"""Deterministic replay gate for the observed DPO task regression."""
import sys,json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--branch',required=True);p.add_argument('--case',choices=['complete','calculate'],default='complete');a=p.parse_args()
import re
if not re.fullmatch(r'[A-Za-z0-9_-]+',a.branch):p.error('safe checkpoint label required')
sys.path[:0]=[str(a.review/'source/src')]
from apt_v2.environment import Environment
from apt_v2.tasks import make_tasks
es=[json.loads(l) for l in (a.review/(a.branch+'-complete-episodes.jsonl')).read_text(encoding='utf8').splitlines()]
e=next(e for e in es if e['task_id']=='development-002' and e['mode']=='greedy');task=next(t for t in make_tasks() if t.id==e['task_id']);env=Environment(task)
try:
    if a.case=='complete':
        for row in e['turns']:
            actual=env.step(row['action'])
            assert actual=={k:v for k,v in row.items() if k not in ('generation','sampling_seed')},'saved action replay differs'
        ok=env.done and env.success;detail={'success':env.success,'turns':len(e['turns']),'termination':env.records[-1]['termination']}
    else:
        # Minimal decision seam: two monthly queries are necessary for calculate_change.
        for row in e['turns'][2:4]:env.step(row['action'])
        assert set(env.verified)=={'current_net_cents','prior_net_cents'}
        action=e['turns'][4]['action'];row=env.step(action)
        ok='change_cents' in env.verified;detail={'action':action,'new_verified_metrics':row['new_verified_metrics']}
    print(json.dumps({'branch':a.branch,'case':a.case,'verdict':'PASS' if ok else 'FAIL',**detail}),flush=True)
    raise SystemExit(0 if ok else 1)
finally:env.close()
