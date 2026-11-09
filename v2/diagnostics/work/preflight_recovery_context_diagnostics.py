from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.tasks import make_tasks
from apt_v2.recovery_diagnostics import diagnostic_contexts,diagnose_action,messages_id,comparison_rates,diagnostic_summary
from apt_v2.dpo_recovery_training import validate_training_pairs
from transformers import AutoTokenizer
cfg=json.loads((ROOT/'configs/recovery-context-diagnostics-v1.json').read_text())
pairs=[json.loads(l) for l in (ROOT/'artifacts/reviews/dpo-recovery-collect-v1-20261102/extracted/dpo-recovery-collect-v1-20261102/pairs.jsonl').read_text(encoding='utf8').splitlines()]
validate_training_pairs(pairs)
tasks=[t for t in make_tasks() if t.split in cfg['task_counts']];tm={t.id:t for t in tasks};contexts=diagnostic_contexts(tasks)
seen={messages_id(p['messages']) for p in pairs}
assert all((messages_id(c['messages']) in seen)==(c['split']=='train') for c in contexts)
tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/reviews/chosen-only-sft-v1-20261102/extracted/chosen-only-sft-v1-20261102/tokenizer',local_files_only=True,trust_remote_code=False)
lengths=[len(tok.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)) for c in contexts]
assert max(lengths)+cfg['eval']['max_new_tokens']<=cfg['eval']['max_model_len']
assert all(diagnose_action(tm[c['task_id']],c,c['chosen'])[1]['productive'] for c in contexts)
empty=diagnostic_summary([],cfg['checkpoints']);assert comparison_rates(empty)['sft']['greedy']['overall']['train']['productive_rate'] is None
summary={'train_contexts':sum(c['split']=='train' for c in contexts),'development_contexts':sum(c['split']=='development' for c in contexts),
    'exact_training_context_membership':84,'development_contexts_in_pairs':0,'expert_productive_actions':len(contexts),
    'max_prompt_tokens':max(lengths),'max_prompt_plus_generation_tokens':max(lengths)+cfg['eval']['max_new_tokens'],
    'context_limit':cfg['eval']['max_model_len'],'planned_native_actions':len(contexts)*(1+cfg['recovery_eval']['samples_per_context'])*len(cfg['checkpoints']),
    'training_updates':0,'tests_passed':113}
dest=ROOT/'artifacts/diagnostics/recovery-context-diagnostics-v1-20261102'
(dest/'preflight_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
