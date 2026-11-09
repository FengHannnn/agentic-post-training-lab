from pathlib import Path
import zipfile,json
src=Path(r'C:/Users/app/Downloads/sft-diagnostics-v1-20261101-review.zip');dest=Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/sft-diagnostics-v1-20261101/extracted');dest.mkdir(exist_ok=True)
with zipfile.ZipFile(src) as z:
    for i in z.infolist():
        if not (dest/i.filename).resolve().is_relative_to(dest.resolve()):raise ValueError('unsafe path')
    z.extractall(dest)
r=dest/'sft-diagnostics-v1-20261101'
read=lambda name:json.loads((r/name).read_text(encoding='utf8'))
for name in ['hf-fp32-probes.json','hf-bf16-probes.json','vllm-probes.json','vllm-base-probes.json','vllm-sft-probes.json']:
    rows=read(name);keys=[(p['mode'],p['probe_id']) for p in rows]
    print(name,'rows',len(rows),'keys',keys)
print('SFT SUMMARY',read('train-sft-summary.json'))
left={(p['mode'],p['probe_id']) for p in read('hf-bf16-probes.json')}
right={(p['mode'],p['probe_id']) for p in read('vllm-probes.json')}
rows=read('vllm-probes.json');probes=read('encoded_probes.json');n=len(probes)
assert len(rows)==2*n
restored=[]
for i,row in enumerate(rows):
    assert row['probe_id']==probes[i%n]['probe_id'] and row['prompt_token_ids']==probes[i%n]['prompt_token_ids']
    restored.append({**row,'sampling_mode':row['mode'],'mode':'base' if i<n else 'sft'})
assert left=={(p['mode'],p['probe_id']) for p in restored}
print('PASS: recovered branch keys match HF; original records preserved')
