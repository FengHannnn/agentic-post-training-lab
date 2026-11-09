from pathlib import Path
import json,sys,hashlib,collections
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.hard_recovery_sft import recovery_weights,HARD_SCENARIOS
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.recovery_diagnostics import diagnostic_contexts,messages_id,diagnose_action
from apt_v2.tasks import make_tasks
from transformers import AutoTokenizer
cfg=json.loads((ROOT/'configs/hard-recovery-sft-weighted-v1.json').read_text())
parent=ROOT/'artifacts/reviews'/cfg['initial_run']/'extracted'/cfg['initial_run']
data=ROOT/'artifacts/reviews'/cfg['data_run']/'extracted'/cfg['data_run']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'final/adapter/adapter_model.safetensors')==cfg['adapter_sha256']
assert sha(data/'pairs.jsonl')==cfg['pairs_sha256']
pairs=[json.loads(l) for l in (data/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
examples=chosen_examples(pairs);weights,summary=recovery_weights(pairs,cfg['hard_weight'])
assert summary['hard_examples']==84 and summary['other_examples']==85 and abs(sum(weights)-169)<1e-10
tok=AutoTokenizer.from_pretrained(parent/'tokenizer',local_files_only=True,trust_remote_code=False)
encoded=encode_chosen(tok,examples,cfg['max_length'])
parent_binding=json.loads((parent/'input_binding.json').read_text())
assert data_sha256(examples)==parent_binding['examples_sha256'] and data_sha256(encoded)==parent_binding['encoded_sha256']
for sample in encoded:
 n=sample['prompt_length'];assert sample['labels'][:n]==[-100]*n and sample['labels'][n:]==sample['input_ids'][n:] and sample['input_ids'][-1]==tok.eos_token_id
tasks=[t for t in make_tasks() if t.split in cfg['task_counts']];contexts=diagnostic_contexts(tasks);seen={messages_id(p['messages']) for p in pairs};tm={t.id:t for t in tasks}
assert all((messages_id(c['messages']) in seen)==(c['split']=='train') for c in contexts)
assert all(diagnose_action(tm[c['task_id']],c,c['chosen'])[1]['productive'] for c in contexts)
dest=ROOT/'artifacts/datasets'/cfg['run_id'];dest.mkdir(parents=True,exist_ok=True)
(dest/'weights.json').write_text(json.dumps({'weights':weights,'summary':summary},ensure_ascii=False,indent=2),encoding='utf8')
(dest/'examples.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in examples),encoding='utf8')
result={'initial_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],'chosen_examples':len(examples),
 'examples_sha256':data_sha256(examples),'encoded_sha256':data_sha256(encoded),'weight_summary':summary,
 'normalized_hard_weight':4/(421/169),'normalized_other_weight':1/(421/169),'supervised_tokens_per_epoch':sum(s['response_length'] for s in encoded),
 'supervised_token_exposures':sum(s['response_length'] for s in encoded)*6,'max_encoded_tokens':max(len(s['input_ids']) for s in encoded),
 'epochs':6,'optimizer_updates':258,'pair_exposures':1014,'train_contexts':84,'development_contexts':28,
 'expert_productive_contexts':112,'planned_recovery_actions':1316,'planned_full_episodes':64,'max_wall_seconds':14400}
(dest/'preflight_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
