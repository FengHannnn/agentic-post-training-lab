from pathlib import Path
import json,sys,collections
r=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/sft-diagnostics-v1-20261101/offline-review/recovered')
sys.path.insert(0,str(r/'source/src'))
from transformers import AutoTokenizer
read=lambda n:json.loads((r/n).read_text(encoding='utf8'))
tok=AutoTokenizer.from_pretrained(r/'tokenizer',local_files_only=True)
probes={p['probe_id']:p for p in read('encoded_probes.json')}
for name in ('hf-fp32-probes.json','hf-bf16-probes.json','vllm-probes.json'):
    for row in read(name):
        if row['mode']=='sft':
            gold=probes[row['probe_id']]['response_token_ids']
            diffs=[{'pos':i,'gold':tok.decode([a]),'argmax':tok.decode([b])} for i,(a,b) in enumerate(zip(gold,row['score']['argmax_token_ids'])) if a!=b]
            print(name,row['probe_id'],'tool',json.loads(probes[row['probe_id']]['response'])['tool'],
                'gold_equal',row['response_token_ids']==gold,'tf_diffs',json.dumps(diffs),'text',row['text'])
rows=[json.loads(l) for l in (r/'hf-fp32-actions.jsonl').read_text(encoding='utf8').splitlines()]
print('SQL_TF_BY_TASK',json.dumps({t:{'exact':sum(x['all_tokens_correct'] for x in rows if x['mode']=='sft' and x['task_id']==t and x['tool']=='query_sql'),
 'total':sum(x['mode']=='sft' and x['task_id']==t and x['tool']=='query_sql' for x in rows)} for t in sorted({x['task_id'] for x in rows})}))
for e in [json.loads(l) for l in (r/'train-sft-episodes.jsonl').read_text(encoding='utf8').splitlines()]:
    sql=[x for x in e['turns'] if x['action'].get('tool')=='query_sql']
    print('TASK',e['task_id'],'sequence',[x['action'].get('tool') for x in e['turns']],
        'queries',[{'sql':x['action']['arguments'].get('sql'),'obs':x['observation']} for x in sql])
