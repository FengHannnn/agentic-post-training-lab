from pathlib import Path
import sys,json,collections,math,re
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.recovery_diagnostics import messages_id
from apt_v2.sft import encode_example
from transformers import AutoTokenizer
run='hard-recovery-sft-weighted-v1-20261103';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()];read=lambda p:json.loads(p.read_text(encoding='utf8'))
episodes={b:{(e['task_id'],e['rollout_index']):e for e in load(out/(b+'-sample-episodes.jsonl'))} for b in ('before','after')}
changes=[];failed=[]
for key,after in episodes['after'].items():
 before=episodes['before'][key]
 def evidence(e):
  return {'success':e['success'],'turns':len(e['turns']),'tools':[r['action'].get('tool','invalid') for r in e['turns']],
   'sql_events':[{'turn':r['turn'],'sql':r['action'].get('arguments',{}).get('sql'),'verified':r['observation'].get('verified_metrics',[]),'error':r['tool_error']} for r in e['turns'] if r['action'].get('tool')=='query_sql'],
   'error_counts':dict(collections.Counter(r['tool_error'] for r in e['turns'] if r['tool_error']))}
 if after['success']!=before['success']:changes.append({'task_id':key[0],'rollout_index':key[1],'before':evidence(before),'after':evidence(after)})
 if not after['success']:failed.append({'task_id':key[0],'rollout_index':key[1],'trajectory':after,'compact':evidence(after)})
actions={b:load(out/(b+'-actions.jsonl')) for b in ('before','after')}
remaining={split:{mode:[{'context_id':r['context_id'],'sample_index':r['sample_index'],'text':r['generation']['text'],'diagnostic':r['diagnostic'],'observation':r['row']['observation']} for r in actions['after'] if r['split']==split and r['mode']==mode and not r['diagnostic']['productive']] for mode in ('greedy','sample')} for split in ('train','development')}
case_result={'paired_full_task_changes':changes,'failed_full_tasks':failed,'remaining_recovery_failures':remaining}
(dest/'behavior_analysis.json').write_text(json.dumps(case_result,ensure_ascii=False,indent=2),encoding='utf8')
for change in changes:
 print('IMPROVED',change['task_id'],change['rollout_index'],json.dumps({'before':{k:v for k,v in change['before'].items() if k!='sql_events'},'after':{k:v for k,v in change['after'].items() if k!='sql_events'},'before_sql_states':[{'verified':x['verified'],'error':x['error']} for x in change['before']['sql_events']],'after_sql_states':[{'verified':x['verified'],'error':x['error']} for x in change['after']['sql_events']]},ensure_ascii=False))
for f in failed:print('REMAINING_FULL_TASK',f['task_id'],f['rollout_index'],json.dumps(f['compact'],ensure_ascii=False))
for split in remaining:
 print('REMAINING_GREEDY',split,json.dumps(remaining[split]['greedy'],ensure_ascii=False))
pairs=load(out/'pairs.jsonl');pre=read(out/'before_chosen_scores.json')['metrics'];post=read(out/'after_chosen_scores.json')['metrics']
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
cm={messages_id(c['messages']):c for c in load(out/'contexts.jsonl') if c['split']=='train'}
g={r['context_id']:r for r in actions['after'] if r['split']=='train' and r['mode']=='greedy'};scored={}
for i,p in enumerate(pairs):
 c=cm.get(messages_id(p['messages']))
 if c is None:continue
 assert json.loads(p['chosen'])==json.loads(c['chosen'])
 enc=encode_example(tok,{'messages':p['messages'],'response':p['chosen']},2048)
 plain=tok(p['chosen'],add_special_tokens=False,return_offsets_mapping=True)
 assert plain['input_ids']==enc['input_ids'][enc['prompt_length']:-1]
 start,end=re.search(r'"tool"\s*:\s*"([^"]+)"',p['chosen']).span(1)
 index=next(j for j,(a,b) in enumerate(plain['offset_mapping']) if a<end and b>start)
 row=g[c['context_id']];assert enc['input_ids'][:enc['prompt_length']]==row['generation']['prompt_token_ids']
 assert plain['input_ids'][:index]==row['generation']['response_token_ids'][:index]
 scored[c['context_id']]={'scenario':c['scenario'],'before':math.exp(pre[i]['token_log_probs'][index]),'after':math.exp(post[i]['token_log_probs'][index]),
  'expected_token':tok.decode([plain['input_ids'][index]]),'greedy_tool_match':row['diagnostic']['tool_matches_expert']}
groups={}
for scenario in sorted({r['scenario'] for r in scored.values()}):
 rows=[r for r in scored.values() if r['scenario']==scenario]
 groups[scenario]={'contexts':len(rows),'expected_token':rows[0]['expected_token'],'before_probability_mean':sum(r['before'] for r in rows)/len(rows),
 'after_probability_mean':sum(r['after'] for r in rows)/len(rows),'greedy_tool_match':sum(r['greedy_tool_match'] for r in rows)}
(dest/'tool_token_analysis.json').write_text(json.dumps({'method':'stored FP32 chosen first tool-value token scores, 84 distinct train contexts; native greedy BF16 separately','groups':groups,'per_context':scored},ensure_ascii=False,indent=2),encoding='utf8')
print('TOOL_TOKEN_SCORES',json.dumps(groups,ensure_ascii=False))
