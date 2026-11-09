from pathlib import Path
import json,sys,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');out=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-collect-v1-20261102/extracted/dpo-recovery-collect-v1-20261102');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-recovery-collect-v1-20261102')
sys.path.insert(0,str(out/'source/src'))
from apt_v2.dpo_recovery import SCENARIOS,assess,encode_recovery_pair
from apt_v2.tasks import make_tasks
from apt_v2.dpo import action_from_text
from transformers import AutoTokenizer
load=lambda name:[json.loads(l) for l in (out/name).read_text(encoding='utf8').splitlines()]
contexts=load('contexts.jsonl');samples=load('samples.jsonl');pairs=load('pairs.jsonl');native=load('native_pairs.jsonl')
cm={c['context_id']:c for c in contexts};tm={t.id:t for t in make_tasks() if t.split=='train'}
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True)
by_scenario={s:{'samples':0,'productive':0,'failed_or_no_progress':0,'incomplete':0,'native_candidates':0,'selected_native':0,'exact_repeat_of_previous_action':0} for s in SCENARIOS}
reasons=collections.Counter();repeated=0;examples={}
for sample in samples:
 c=cm[sample['context_id']];d=by_scenario[c['scenario']];d['samples']+=1
 row,reason=assess(tm[c['task_id']],c,sample['text'])
 complete=sample['finish_reason']=='stop' and sample['response_token_ids'][-1]==tok.eos_token_id
 if not complete:d['incomplete']+=1
 if reason:d['failed_or_no_progress']+=1;reasons[reason]+=1
 else:d['productive']+=1
 if action_from_text(sample['text'])==c['records'][-1]['row']['action']:
  d['exact_repeat_of_previous_action']+=1;repeated+=1
for p in native:by_scenario[p['scenario']]['native_candidates']+=1
for p in pairs:
 if p.get('rejected_source')=='frozen_dpo_sample':by_scenario[p['scenario']]['selected_native']+=1
 if 'scenario' in p and (p['scenario'] not in examples or p['rejected_source']=='frozen_dpo_sample'):
  examples[p['scenario']]={k:p[k] for k in ('task_id','scenario','chosen','rejected','reason','rejected_source','chosen_observation','rejected_observation')}
encoded=[encode_recovery_pair(tok,p,2048) for p in pairs]
result={'by_scenario':by_scenario,'failure_reasons':dict(reasons),'exact_repeat_actions':repeated,
 'native_candidate_reasons':dict(collections.Counter(p['reason'] for p in native)),
 'selected_native_chosen_tools':dict(collections.Counter(action_from_text(p['chosen']).get('tool') for p in pairs if p['rejected_source']=='frozen_dpo_sample')),
 'max_pair_tokens':max(len(e['prompt'])+max(len(e['chosen']),len(e['rejected'])) for e in encoded),
 'tokens_by_source':dict((source,sum(len(e['chosen'])+len(e['rejected']) for p,e in zip(pairs,encoded) if p['rejected_source']==source)) for source in sorted({p['rejected_source'] for p in pairs}))}
(dest/'recovery_behavior_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
(dest/'representative_pairs.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2));print('EXAMPLES',json.dumps(examples,ensure_ascii=False))
