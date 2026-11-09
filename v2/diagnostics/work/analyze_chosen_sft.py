from pathlib import Path
import json,collections
R=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102');load=lambda n:[json.loads(x) for x in (R/n).read_text(encoding='utf8').splitlines() if x];read=lambda n:json.loads((R/n).read_text(encoding='utf8'))
actions={b:load(b+'-recovery-actions.jsonl') for b in ('before','dpo','after')};contexts={c['context_id']:c for c in load('recovery_contexts.jsonl')};details=[]
for x in actions['after']:
 if x['scenario']=='failed_check_repair' and x['result']['productive']:
  ctx=contexts[x['context_id']];details.append({k:x[k] for k in ('context_id','sample_index','sample_seed') }|{'previous_action':ctx['records'][-1]['row']['action'],'previous_observation':ctx['records'][-1]['row']['observation'],'action':json.loads(x['generation']['text']),'observation':x['result']['row']['observation']})
paired={}
for scenario in sorted({x['scenario'] for x in actions['after']}):
 maps={b:{(x['context_id'],x['sample_index']):x for x in rows if x['mode']=='sample' and x['scenario']==scenario} for b,rows in actions.items()}
 paired[scenario]=dict(collections.Counter(('both_productive' if maps['dpo'][k]['result']['productive'] else 'improved') if v['result']['productive'] else ('regressed' if maps['dpo'][k]['result']['productive'] else 'both_failed') for k,v in maps['after'].items()))
patterns={}
for b,rows in actions.items():
 patterns[b]={}
 for s in sorted({x['scenario'] for x in rows}):
  group=[x for x in rows if x['scenario']==s and x['mode']=='sample']
  patterns[b][s]={'tools':dict(collections.Counter(x['result']['row']['action'].get('tool','invalid') for x in group)),'errors':dict(collections.Counter(x['result']['row']['tool_error'] or x['result']['reason'] or 'productive' for x in group))}
pairs=load('pairs.jsonl');pre=read('before_chosen_scores.json');post=read('after_chosen_scores.json');groups={}
for group in sorted({p.get('scenario','original_sql') for p in pairs}):
 idx=[i for i,p in enumerate(pairs) if p.get('scenario','original_sql')==group]
 groups[group]={'rows':len(idx),'before_mean_ce':sum(pre['metrics'][i]['loss'] for i in idx)/len(idx),'after_mean_ce':sum(post['metrics'][i]['loss'] for i in idx)/len(idx)}
result={'successful_failed_check_repairs':details,'paired_dpo_to_sft_recovery':paired,'patterns':patterns,'chosen_ce_by_scenario':groups}
(R.parent.parent/'behavior_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'cases':details,'paired':paired,'after_patterns':patterns['after'],'ce_groups':groups},ensure_ascii=False,indent=2))
