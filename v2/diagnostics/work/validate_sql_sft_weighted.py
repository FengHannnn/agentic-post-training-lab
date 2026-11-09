from pathlib import Path
import sys,json,zipfile,torch
ROOT=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2');sys.path.insert(0,str(ROOT/'src'))
from apt_v2.sql_sft import normalized_weights,validate_examples
from apt_v2.sft import encode_example
from transformers import AutoTokenizer
archive=ROOT/'artifacts/reviews/sft-pilot-v1-20261030/sft-pilot-v1-20261030-review.zip'
with zipfile.ZipFile(archive) as z:
    examples=[json.loads(l) for l in z.read('sft-pilot-v1-20261030/examples.jsonl').decode().splitlines()]
    summary=json.loads(z.read('sft-pilot-v1-20261030/data_summary.json'))
    import io
    state=torch.load(io.BytesIO(z.read('sft-pilot-v1-20261030/final/state.pt')),map_location='cpu',weights_only=True)
counts=validate_examples(examples,summary);weights=normalized_weights(examples,4)
assert {int(s['step']) for s in state['optimizer']['state'].values()}=={63}
tok=AutoTokenizer.from_pretrained(r'C:/Users/app/Documents/Codex/2026-10-28/new-chat/work/reviews/baseline-1.5b-20261029-01/tokenizer',local_files_only=True)
samples=[encode_example(tok,e,2048) for e in examples]
result={'examples':len(examples),'tool_counts':counts,'dataset_sha256':summary['dataset_sha256'],
 'max_tokens':max(len(s['input_ids']) for s in samples),'parent_adam_steps':63,'weight_mean':sum(weights)/len(weights),
 'supervised_tokens':sum(s['response_length'] for s in samples)}
print(json.dumps(result))
(ROOT/'reports/sql-sft-weighted-v1-data-check-20261101.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
