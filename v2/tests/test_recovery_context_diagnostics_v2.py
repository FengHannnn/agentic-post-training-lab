import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.tasks import make_tasks
from apt_v2.dpo import pair_id
from apt_v2.dpo_recovery import build_context,SCENARIOS
from apt_v2.recovery_diagnostics import diagnose_action,summarize,diagnostic_contexts,diagnostic_summary,messages_id
from test_dpo_v2 import CharTokenizer
from test_dpo_recovery_training_v2 import pair_fixtures

def context(scenario):
    task=next(t for t in make_tasks() if t.split=='development')
    return task,build_context(task,scenario,allowed_split='development')

def test_missing_verified_metric_is_a_tool_choice_error_before_an_id_error():
    task,c=context('sql_unverified_requery')
    row,d=diagnose_action(task,c,c['scripted_rejected'])
    assert d['expected_tool']=='query_sql' and d['actual_tool']=='calculate_change' and not d['tool_matches_expert']
    assert not d['evidence']['all_required_metrics_available'] and not d['productive']
    expert=json.loads(c['chosen']);row,d=diagnose_action(task,c,json.dumps(expert))
    assert d['productive'] and d['tool_matches_expert'] and d['sql']['status']=='verified_target'
    assert d['sql']['target_verified'] and 'current_net_cents' in row['new_verified_metrics']

def test_existing_empty_evidence_and_wrong_metric_ids_are_separate_categories():
    task,c=context('repeat_evidence_error');_,d=diagnose_action(task,c,c['scripted_rejected'])
    assert [s['status'] for s in d['evidence']['slots']]==['id_exists_unverified','correct_verified_metric']
    task,c=context('stale_evidence_repair');action=json.loads(c['chosen'])
    _,good=diagnose_action(task,c,c['chosen'])
    assert good['evidence']['all_required_metrics_available'] and good['evidence']['all_references_verified']
    action['arguments']['current_evidence'],action['arguments']['prior_evidence']=action['arguments']['prior_evidence'],action['arguments']['current_evidence']
    _,bad=diagnose_action(task,c,json.dumps(action))
    assert all(s['status']=='id_exists_wrong_metric' for s in bad['evidence']['slots'])
    assert bad['evidence']['all_required_metrics_available'] and not bad['productive']
    action['arguments']={'current_evidence':[],'prior_evidence':'missing'}
    _,bad=diagnose_action(task,c,json.dumps(action))
    assert [s['status'] for s in bad['evidence']['slots']]==['invalid_reference_type','unknown_id']

def test_sql_execution_goal_validation_and_no_progress_have_different_verdicts():
    from apt_v2.tasks import metric_sql
    task,c=context('sql_unverified_requery')
    for sql,status in [('SELECT current_net_cents FROM metrics','execution_error'),
        ('SELECT SUM(gross_cents) AS current_net_cents FROM orders','executed_unverified'),
        (metric_sql(task,'prior_net_cents'),'verified_other_metric')]:
        _,d=diagnose_action(task,c,json.dumps({'tool':'query_sql','arguments':{'sql':sql}}))
        assert d['sql']['status']==status and d['sql']['target_verified'] is False
    task,c=context('failed_check_repair');_,bad=diagnose_action(task,c,c['scripted_rejected'])
    assert bad['evidence']['all_references_verified'] and bad['failed_checks']==['change_cents'] and not bad['productive']
    _,redundant=diagnose_action(task,c,json.dumps({'tool':'query_sql','arguments':{'sql':metric_sql(task,'current_net_cents')}}))
    assert redundant['sql']['status']=='verified_any' and redundant['reason']=='no_progress' and not redundant['tool_matches_expert']
    summary=summarize([{'diagnostic':bad},{'diagnostic':redundant}])
    assert summary['evidence']['actions']==1 and summary['evidence']['all_references_verified']==1
    assert summary['sql']['any_metric_verified']==1 and summary['productive']==0

def test_train_development_native_replay_membership_and_tamper_detection(tmp_path,monkeypatch):
    import diagnose_v2_recovery_contexts as worker
    tasks=[next(t for t in make_tasks() if t.split==s) for s in ('train','development')]
    monkeypatch.setattr(worker,'make_tasks',lambda:tasks)
    cfg={'task_counts':{'train':1,'development':1},'checkpoints':{'before':{},'dpo':{},'sft':{}},
        'recovery_eval':{'seed':53,'split_seed_offsets':{'train':1000000,'development':0},'samples_per_context':3,'temperature':.8},
        'eval':{'max_new_tokens':100000}}
    contexts=diagnostic_contexts(tasks);tok=CharTokenizer();tm={t.id:t for t in tasks}
    pairs=pair_fixtures()
    for i,c in enumerate([c for c in contexts if c['split']=='train']):
        pairs[i].update(messages=c['messages']);pairs[i]['pair_id']=pair_id(pairs[i])
    seen={messages_id(p['messages']) for p in pairs}
    membership=[{'context_id':c['context_id'],'task_id':c['task_id'],'split':c['split'],'messages_sha256':messages_id(c['messages']),
        'seen_in_training_pairs':messages_id(c['messages']) in seen} for c in contexts]
    worker.jsonl(tmp_path/'contexts.jsonl',contexts);worker.jsonl(tmp_path/'context_membership.jsonl',membership);worker.jsonl(tmp_path/'pairs.jsonl',pairs)
    allrows=[]
    for branch in cfg['checkpoints']:
        rows=[]
        for c in contexts:
            for mode in ('greedy','sample'):
                for ri in range(1 if mode=='greedy' else 3):
                    text=c['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
                    g={'text':text,'mode':'greedy' if mode=='greedy' else 'dpo_sample','prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),
                        'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),'finish_reason':'stop'}
                    actual,d=diagnose_action(tm[c['task_id']],c,text)
                    rows.append({'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':c['split'],'context_id':c['context_id'],
                        'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':worker.sampling_seed(cfg,tasks,c,ri),
                        'temperature':0 if mode=='greedy' else .8,'generation':g,'row':actual,'diagnostic':d})
        worker.jsonl(tmp_path/(branch+'-actions.jsonl'),rows);allrows.extend(rows)
    summary=diagnostic_summary(allrows,cfg['checkpoints']);(tmp_path/'diagnostic_summary.json').write_text(json.dumps(summary))
    evidence=worker.audit_diagnostics(tmp_path,cfg,tok)
    assert evidence['actions_replayed']==168 and evidence['train_contexts']==7 and evidence['development_contexts']==7 and evidence['training_contexts_exactly_in_pairs']==7
    assert summary['sft']['train']['sample']['overall']['sql']['target_verified_when_required']==9
    worker.export_csv(tmp_path)
    assert len((tmp_path/'diagnostic_counts.csv').read_text(encoding='utf-8-sig').splitlines())==97
    rows[0]['diagnostic']['tool_matches_expert']=False
    (tmp_path/'sft-actions.jsonl').unlink();worker.jsonl(tmp_path/'sft-actions.jsonl',rows)
    with pytest.raises(ValueError,match='replay'):worker.audit_diagnostics(tmp_path,cfg,tok)
    rows[0]['diagnostic']['tool_matches_expert']=True;rows[0]['sample_seed']+=1
    (tmp_path/'sft-actions.jsonl').unlink();worker.jsonl(tmp_path/'sft-actions.jsonl',rows)
    with pytest.raises(ValueError,match='identity'):worker.audit_diagnostics(tmp_path,cfg,tok)

def test_diagnostic_runner_failure_keeps_live_log_and_review_zip(tmp_path,monkeypatch,capsys):
    import run_v2_recovery_context_diagnostics as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir()
    for name in ('audit.json','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        p=tmp_path/'artifacts/initial'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    p=tmp_path/'artifacts/data/pairs.jsonl';p.parent.mkdir(parents=True);p.write_text('fixture')
    cfg={'run_id':'diagnostic','checkpoints':{'before':{'run':'initial'}},'data_run':'data','max_wall_seconds':14400}
    (tmp_path/'configs/recovery-context-diagnostics-v1.json').write_text(json.dumps(cfg))
    (tmp_path/'recovery_context_diagnostic_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/diagnose_v2_recovery_contexts.py').write_text("print('diagnostic live progress',flush=True)\nraise RuntimeError('intentional')")
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    assert 'diagnostic live progress' in capsys.readouterr().out
    import zipfile
    with zipfile.ZipFile(tmp_path/'diagnostic-review.zip') as z:assert 'diagnostic/prepare.log' in z.namelist()

def test_configuration_pins_three_existing_models_and_keeps_generation_settings():
    cfg=json.loads((ROOT/'configs/recovery-context-diagnostics-v1.json').read_text())
    previous=json.loads((ROOT/'configs/chosen-only-sft-v1.json').read_text())
    assert cfg['checkpoints']['before']['adapter_sha256']==previous['adapter_sha256']
    assert cfg['checkpoints']['dpo']['adapter_sha256']==previous['comparison_adapter_sha256']
    assert cfg['checkpoints']['sft']['adapter_sha256']=='078182d293d37b0caeda3bdef8355d4be45439af5480228259ef8658ff868a55'
    assert cfg['algorithm']=='inference_only_recovery_diagnostic' and cfg['max_wall_seconds']==14400
    assert cfg['eval']=={k:v for k,v in previous['eval'].items() if k in ('max_model_len','max_new_tokens','gpu_memory_utilization')}
    assert cfg['recovery_eval']['seed']==previous['recovery_eval']['seed']
