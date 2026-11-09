from pathlib import Path
import json,sys,re
r=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101')
sys.path.insert(0,str(r/'source/src'))
from apt_v2.tasks import make_tasks,metric_sql
byid={t.id:t for t in make_tasks()};valid=exact=space_equal=0;faults=[];submits=checks=0
for split in ('train','development'):
    episodes=[json.loads(l) for l in (r/('weighted-'+split+'-episodes.jsonl')).read_text(encoding='utf8').splitlines()]
    for e in episodes:
        for row in e['turns']:
            a=row['action'];o=row['observation']
            if a['tool']=='query_sql' and o.get('verified_metrics'):
                m=o['verified_metrics'][0];gold=metric_sql(byid[e['task_id']],m);s=a['arguments']['sql']
                valid+=1;exact+=s==gold;space_equal+=re.sub(r'\s+','',s)==re.sub(r'\s+','',gold)
            if a['tool']=='check_result' and o.get('passed'):checks+=1
            if a['tool']=='submit' and o.get('passed'):submits+=1
        if e['recovered']:faults.append({'task':e['task_id'],'tools':[x['action']['tool'] for x in e['turns']]})
print(json.dumps({'valid_sql':valid,'exact_expert_sql':exact,'equal_after_removing_whitespace':space_equal,'valid_checks':checks,'valid_submits':submits,'fault_examples':faults[:2]}))
