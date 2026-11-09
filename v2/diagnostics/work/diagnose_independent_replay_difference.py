from pathlib import Path
import json,sys,sqlite3
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');run='hard-recovery-independent-test-v1-20261103';out=ROOT/'artifacts/reviews'/run/'extracted'/run
sys.path[:0]=[str(out/'source/src')]
from apt_v2.tasks import make_tasks
from apt_v2.environment import Environment
from apt_v2.dpo import action_from_text
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
tm={t.id:t for t in make_tasks()};diffs=[]
for b in ('before','after'):
 for suffix in ('test','sample'):
  for e in load(out/(b+'-'+suffix+'-episodes.jsonl')):
   env=Environment(tm[e['task_id']],e['reward_mode'])
   try:
    for r in e['turns']:
     a=env.step(action_from_text(r['generation']['text']))
     changes={k:{'local':a[k],'cloud':r[k]} for k in ('action','observation','reward','done','success','tool_error','process_reward','outcome_reward','new_verified_metrics','termination') if a[k]!=r[k]}
     if changes:diffs.append({'branch':b,'mode':suffix,'task':e['task_id'],'rollout':e.get('rollout_index'),'turn':r['turn'],'text':r['generation']['text'],'changes':changes})
   finally:env.close()
print(json.dumps({'sqlite':sqlite3.sqlite_version,'diffs':diffs},ensure_ascii=False,indent=2))
(ROOT/'artifacts/reviews'/run/'platform_replay_differences.json').write_text(json.dumps(diffs,ensure_ascii=False,indent=2),encoding='utf8')
