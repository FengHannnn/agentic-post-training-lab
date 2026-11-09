from pathlib import Path
import json,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2')
run='recovery-context-diagnostics-v1-20261102';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
summary=json.loads((out/'diagnostic_summary.json').read_text());allrows={b:load(out/(b+'-actions.jsonl')) for b in summary}
for b in summary:
 for split in ('train','development'):
  for mode in ('greedy','sample'):
   print(b,split,mode)
   for scenario,s in summary[b][split][mode]['by_scenario'].items():
    e=s['evidence'];q=s['sql']
    print(scenario,json.dumps({'productive':[s['productive'],s['actions']],'tools_match':s['tool_matches_expert'],'actual_tools':s['actual_tool_counts'],
    'ids_on_expected_tool':[e['expected_tool_all_references_verified'],e['expected_tool_actions']],
    'sql':[q['query_chosen_when_required'],q['query_required_actions'],q['target_verified_when_required']]}))
cases={}
for b in allrows:
 cases[b]={}
 for scenario in ('sql_unverified_requery','repeat_evidence_error','repeat_sql_error','failed_check_repair'):
  rows=[r for r in allrows[b] if r['split']=='train' and r['scenario']==scenario and r['mode']=='greedy']
  cases[b][scenario]=[{'context_id':r['context_id'],'text':r['generation']['text'],'tool_error':r['row']['tool_error'],'observation':r['row']['observation'],'failed_checks':r['diagnostic']['failed_checks']} for r in rows[:2]]
result={'train_greedy_cases':cases,'finish_length_cases':[{'branch':b,'context_id':r['context_id'],'mode':r['mode'],'sample_index':r['sample_index']} for b,rows in allrows.items() for r in rows if r['generation']['finish_reason']=='length']}
(dest/'behavior_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print('CASES',json.dumps(result,ensure_ascii=False))
