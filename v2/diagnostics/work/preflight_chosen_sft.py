from pathlib import Path
import sys,json,hashlib
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path.insert(0,str(ROOT/'src'))
from transformers import AutoTokenizer
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.dpo_recovery import encode_recovery_pair
source=ROOT/'artifacts/datasets/dpo-recovery-v1-20261102/pairs.jsonl'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='c7606a3fbf7ac4437d1aa8b3277228a7b4cdae95106e5d515d55c8f72869e329'
pairs=[json.loads(l) for l in source.read_text(encoding='utf8').splitlines()];examples=chosen_examples(pairs)
tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/reviews/dpo-train-v1-20261102/extracted/dpo-train-v1-20261102/tokenizer',local_files_only=True)
encoded=encode_chosen(tok,examples,2048)
for pair,sample in zip(pairs,encoded):
 e=encode_recovery_pair(tok,pair,2048)
 assert sample['input_ids']==e['prompt']+e['chosen'] and sample['labels']==[-100]*len(e['prompt'])+e['chosen']
unique=len({data_sha256({'messages':e['messages'],'response':e['response']}) for e in examples})
result={'chosen_examples':len(examples),'unique_context_chosen_texts':unique,'sampling_weights_preserved':True,'examples_sha256':data_sha256(examples),
'encoded_sha256':data_sha256(encoded),'supervised_tokens_per_epoch':sum(e['response_length'] for e in encoded),
'max_chosen_context_tokens':max(len(e['input_ids']) for e in encoded),'chosen_tokens_match_dpo_inputs':True,'prompt_labels_all_masked':True}
out=ROOT/'artifacts/datasets/chosen-only-sft-v1-20261102';out.mkdir(parents=True,exist_ok=True)
(out/'examples.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in examples),encoding='utf8')
(out/'preflight_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
