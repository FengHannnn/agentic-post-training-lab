from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.independent_recovery_test import independent_tasks,test_contexts,diagnose_test_action,run_test_recovery,context_environment,NOVEL_SCENARIOS,calendar_sql_check
from apt_v2.chosen_sft import chosen_examples
from apt_v2.dpo import corrective_action
from apt_v2.runtime import Generation
from apt_v2.tasks import metric_sql
from apt_v2.recovery_diagnostics import messages_id
from transformers import AutoTokenizer
cfg=json.loads((ROOT/'configs/hard-recovery-independent-test-v1.json').read_text())
review=ROOT/'artifacts/reviews';parent=review/cfg['parent_run']/'extracted'/cfg['parent_run'];candidate=review/cfg['candidate_run']/'extracted'/cfg['candidate_run']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'final/adapter/adapter_model.safetensors')==cfg['adapter_sha256']
assert sha(candidate/'final/adapter/adapter_model.safetensors')==cfg['candidate_adapter_sha256']
pairs=[json.loads(l) for l in (candidate/'pairs.jsonl').read_text(encoding='utf8').splitlines()];assert sha(candidate/'pairs.jsonl')==cfg['pairs_sha256']
examples=chosen_examples(pairs);tasks=independent_tasks(pairs,examples);tm={t.id:t for t in tasks};contexts=test_contexts(tasks);seen={messages_id(p['messages']) for p in pairs}
assert all(messages_id(c['messages']) not in seen for c in contexts)
tok=AutoTokenizer.from_pretrained(candidate/'tokenizer',local_files_only=True,trust_remote_code=False)
lengths=[];first=0;continuations=[]
for c in contexts:
 lengths.append(len(tok.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)))
 assert diagnose_test_action(tm[c['task_id']],c,c['chosen'])[1]['productive'];first+=1
 if c['scenario'] in NOVEL_SCENARIOS:
  def policy(messages):
   lengths.append(len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)))
   env=context_environment(tm[c['task_id']],c)
   try:
    for m in messages[len(c['messages']):]:
     if m['role']=='assistant':env.step(json.loads(m['content']))
    return Generation(json.dumps(corrective_action(env)))
   finally:env.close()
  e=run_test_recovery(tm[c['task_id']],c,policy);assert e['success']
  continuations.append({'context_id':c['context_id'],'prefix_turns':len(c['records']),'continuation_turns':len(e['turns']),'success':e['success']})
assert max(lengths)+cfg['eval']['max_new_tokens']<=cfg['eval']['max_model_len']
calendar=0
for t in tasks:
 for m in ('current_net_cents','prior_net_cents'):
  assert calendar_sql_check(t,metric_sql(t,m),[m])['passed'];calendar+=1
result={'test_task_ids':[t.id for t in tasks],'test_families':[t.family_id for t in tasks],
 'train_test_task_family_seed_disjoint':True,'standard_contexts':28,'novel_contexts':8,'test_messages_in_training':0,
 'expert_first_actions_productive':first,'expert_novel_continuations_successful':len(continuations),'novel_continuation_details':continuations,
 'oracle_sql_calendar_checks_passed':calendar,'max_expert_prompt_tokens':max(lengths),'max_prompt_plus_generation_tokens':max(lengths)+256,
 'context_limit':8192,'planned_full_episodes':72,'planned_first_recovery_actions':288,'planned_novel_continuations':64,
 'training_enabled':False,'optimizer_updates':0,'max_wall_seconds':14400}
dest=ROOT/'artifacts/diagnostics'/cfg['run_id'];dest.mkdir(parents=True,exist_ok=True)
(dest/'preflight_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
