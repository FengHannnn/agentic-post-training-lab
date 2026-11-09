from pathlib import Path
import json,sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from apt_v2.tasks import make_tasks
from apt_v2.environment import Environment
from apt_v2.runtime import parse_action
z=zipfile.ZipFile(ROOT/'artifacts/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030-review.zip')
e=json.loads(z.read('sft-pilot-v1-20261030/evaluation.jsonl').decode().splitlines()[0])
task=next(t for t in make_tasks() if t.id==e['task_id'])
env=Environment(task)
try:
    for row in e['turns'][:3]:
        actual=env.step(parse_action(row['generation']['text']))
        assert actual['observation']==row['observation']
    print(json.dumps({'task':task.id,'sql':e['turns'][2]['action']['arguments']['sql'],
        'verified_metrics':actual['observation'].get('verified_metrics')}))
    assert 'current_net_cents' in actual['observation'].get('verified_metrics',[]),'reproduced: first monthly query fails business verification'
finally:env.close()
