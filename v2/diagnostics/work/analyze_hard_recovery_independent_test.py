from pathlib import Path
import json,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');run='hard-recovery-independent-test-v1-20261103';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
res={'complete_failures':{},'novel':{},'standard':{},'length_finished':[],'failed_checks':[]}
novel={'both_sql_unverified','both_sql_unverified_repeated_evidence'}
for b in ('before','after'):
 es=load(out/(b+'-sample-episodes.jsonl'));fail=[]
 for e in es:
  if not e['success']:
   fail.append({'task':e['task_id'],'rollout':e['rollout_index'],'termination':e['turns'][-1]['termination'],'actions':[{'turn':r['turn'],'tool':r['action'].get('tool'),'error':r['tool_error'],'verified':r['new_verified_metrics'],'passed':r['observation'].get('passed'),'text':r['generation']['text'] if r['action'].get('tool')=='query_sql' else None} for r in e['turns']]})
  for r in e['turns']:
   if r['generation']['finish_reason']=='length':res['length_finished'].append({'branch':b,'kind':'complete_sample','task':e['task_id'],'rollout':e['rollout_index'],'turn':r['turn'],'tokens':len(r['generation']['response_token_ids']),'tool':r['action'].get('tool')})
 res['complete_failures'][b]=fail
 rows=load(out/(b+'-recovery-actions.jsonl'));eps=load(out/(b+'-recovery-episodes.jsonl'))
 for mode in ('greedy','sample'):
  for name,in_novel in [('novel',True),('standard',False)]:
   group=[r for r in rows if r['mode']==mode and (r['scenario'] in novel)==in_novel]
   res[name][b+'_'+mode]={'actions':len(group),'productive':sum(r['diagnostic']['productive'] for r in group),'tools':dict(collections.Counter(r['row']['action'].get('tool','invalid') for r in group)),
    'repeated_previous_action':sum(r['diagnostic']['repeated_previous_action'] for r in group),'sql_status':dict(collections.Counter(r['diagnostic']['sql']['status'] for r in group if r['diagnostic']['sql']))}
  selected=[e for e in eps if e['mode']==mode]
  first=[r for r in rows if r['mode']==mode and r['scenario'] in novel]
  res['novel'][b+'_'+mode].update({'continuation_success':sum(e['success'] for e in selected),'continuation_tool_errors':sum(e['tool_errors'] for e in selected),'any_verified_in_continuation':sum(any(r['new_verified_metrics'] for r in e['turns']) for e in selected),
    'first_examples':[{'context':r['context_id'],'sample':r['sample_index'],'text':r['generation']['text'],'observation':r['row']['observation']} for r in first[:2]],
    'continuation_errors':dict(collections.Counter(r['tool_error'] for e in selected for r in e['turns'] if r['tool_error']))})
 for e in eps:
  for r in e['turns']:
   if r['generation']['finish_reason']=='length':res['length_finished'].append({'branch':b,'kind':'novel_'+e['mode'],'context':e['context_id'],'sample':e['sample_index'],'turn':r['turn'],'tokens':len(r['generation']['response_token_ids']),'tool':r['action'].get('tool')})
 for r in rows:
  if r['scenario']=='failed_check_repair' and r['mode']=='greedy':res['failed_checks'].append({'branch':b,'task':r['task_id'],'text':r['generation']['text'],'observation':r['row']['observation']})
(dest/'behavior_analysis.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(res,ensure_ascii=False,indent=2))
