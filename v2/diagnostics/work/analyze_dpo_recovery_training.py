from pathlib import Path
import json,collections
out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102/extracted/dpo-recovery-train-v1-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-train-v1-20261102')
load=lambda name:[json.loads(l) for l in (out/name).read_text(encoding='utf8').splitlines()]
left=load('before-sample-episodes.jsonl');right=load('after-sample-episodes.jsonl');changes=[]
for index in (1,2):
 a=next(e for e in left if e['task_id']=='development-000' and e['rollout_index']==index)
 b=next(e for e in right if e['task_id']=='development-000' and e['rollout_index']==index)
 first=next((i for i,(x,y) in enumerate(zip(a['turns'],b['turns'])) if x['generation']['text']!=y['generation']['text']),None)
 brief=lambda e:[{'turn':r['turn'],'action':r['action'],'tool_error':r['tool_error'],'new_verified_metrics':r['new_verified_metrics']} for r in e['turns']]
 changes.append({'task_id':a['task_id'],'rollout_index':index,'first_divergence':first,'before':brief(a),'after':brief(b)})
greedy_identical=all([r['generation']['text'] for r in a['turns']]==[r['generation']['text'] for r in b['turns']] for split in ('train','development') for a,b in zip(load('before-'+split+'-episodes.jsonl'),load('after-'+split+'-episodes.jsonl')))
recovery=[]
for branch in ('before','after'):
 for scenario in ('sql_unverified_requery','repeat_evidence_error','repeat_sql_error','failed_check_repair'):
  rows=[r for r in load(branch+'-recovery-actions.jsonl') if r['mode']=='greedy' and r['scenario']==scenario]
  recovery.append({'branch':branch,'scenario':scenario,'actions':dict(collections.Counter(r['result']['row']['action'].get('tool','invalid') for r in rows)),
   'example':{k:rows[0][k] for k in ('task_id','generation','result')}})
(dest/'behavior_trace_analysis.json').write_text(json.dumps({'greedy_full_task_actions_identical':greedy_identical,'improved_task_traces':changes,'critical_recovery_examples':recovery},ensure_ascii=False,indent=2),encoding='utf8')
print('greedy_identical',greedy_identical)
for c in changes:
 print('CHANGE',c['task_id'],c['rollout_index'],'first',c['first_divergence'])
 for branch in ('before','after'):print(branch,json.dumps(c[branch],ensure_ascii=False))
for r in recovery:print('RECOVERY',r['branch'],r['scenario'],r['actions'],'example_action',json.dumps(r['example']['result']['row']['action'],ensure_ascii=False))
