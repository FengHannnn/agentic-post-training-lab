from pathlib import Path
import json,re,math,sys,collections
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path.insert(0,str(ROOT/'src'))
from apt_v2.recovery_diagnostics import messages_id
from apt_v2.sft import encode_example
from transformers import AutoTokenizer
run='recovery-context-diagnostics-v1-20261102';dest=ROOT/'artifacts/reviews'/run;out=dest/'extracted'/run
parent=ROOT/'artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102'
read=lambda p:json.loads(p.read_text(encoding='utf8'));load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
pairs=load(parent/'pairs.jsonl');pre=read(parent/'before_chosen_scores.json')['metrics'];post=read(parent/'after_chosen_scores.json')['metrics']
tok=AutoTokenizer.from_pretrained(parent/'tokenizer',local_files_only=True,trust_remote_code=False)
context_map={messages_id(c['messages']):c for c in load(out/'contexts.jsonl') if c['split']=='train'}
greedy={r['context_id']:r for r in load(out/'sft-actions.jsonl') if r['split']=='train' and r['mode']=='greedy'}
rows=[]
for i,p in enumerate(pairs):
 c=context_map.get(messages_id(p['messages']))
 if c is None:continue
 enc=encode_example(tok,{'messages':p['messages'],'response':p['chosen']},2048)
 response=enc['input_ids'][enc['prompt_length']:]
 plain=tok(p['chosen'],add_special_tokens=False,return_offsets_mapping=True)
 assert plain['input_ids']==response[:-1]
 start,end=re.search(r'"tool"\s*:\s*"([^"]+)"',p['chosen']).span(1)
 indices=[j for j,(a,b) in enumerate(plain['offset_mapping']) if a<end and b>start]
 first=indices[0];r=greedy[c['context_id']];native=r['generation']['response_token_ids']
 assert enc['input_ids'][:enc['prompt_length']]==r['generation']['prompt_token_ids']
 prefix_equal=native[:first]==response[:first]
 rows.append({'context_id':c['context_id'],'scenario':c['scenario'],'pair_index':i,'expected_tool':json.loads(p['chosen'])['tool'],
  'greedy_tool':r['diagnostic']['actual_tool'],'first_tool_token':tok.decode([response[first]]),'first_tool_token_index':first,
  'correct_first_tool_token_probability_before':math.exp(pre[i]['token_log_probs'][first]),
  'correct_first_tool_token_probability_after':math.exp(post[i]['token_log_probs'][first]),
  'greedy_and_chosen_prefix_match_before_tool':prefix_equal,
  'native_greedy_first_tool_token':tok.decode([native[first]]) if prefix_equal else None,
  'native_greedy_token_probability':math.exp(r['generation']['raw_token_log_probs'][first]) if prefix_equal else None})
unique={r['context_id']:r for r in rows};groups={}
for s in sorted({r['scenario'] for r in unique.values()}):
 group=[r for r in unique.values() if r['scenario']==s]
 groups[s]={'contexts':len(group),'before_expected_first_tool_token_probability_mean':sum(r['correct_first_tool_token_probability_before'] for r in group)/len(group),
  'after_expected_first_tool_token_probability_mean':sum(r['correct_first_tool_token_probability_after'] for r in group)/len(group),
  'after_expected_first_tool_token_probability_range':[min(r['correct_first_tool_token_probability_after'] for r in group),max(r['correct_first_tool_token_probability_after'] for r in group)],
  'greedy_tool_match':sum(r['expected_tool']==r['greedy_tool'] for r in group),
  'prefixes_match_before_tool':sum(r['greedy_and_chosen_prefix_match_before_tool'] for r in group)}
result={'method':'stored FP32 teacher-forced chosen scores vs current BF16 native greedy; first token overlapping JSON tool value; 84 unique train contexts','groups':groups,'per_context':list(unique.values())}
(dest/'tool_token_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(groups,ensure_ascii=False,indent=2))
print('CASES',json.dumps([r for r in unique.values() if r['scenario'] in ('sql_unverified_requery','repeat_evidence_error','repeat_sql_error')][:4],ensure_ascii=False))
