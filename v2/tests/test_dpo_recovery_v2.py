import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation,parse_action
from apt_v2.dpo_recovery import SCENARIOS,build_context,assess,make_pair,encode_recovery_pair,select_pairs
from test_dpo_v2 import CharTokenizer

def test_all_train_recovery_contexts_have_productive_expert_and_failed_negative():
    tok=CharTokenizer();tools={};count=0
    for task in make_tasks()[:12]:
        for scenario in SCENARIOS:
            context=build_context(task,scenario);pair=make_pair(task,context,tok,100000)
            assert pair is not None and pair['rejected_source']=='scripted_failure'
            assert pair['rejected_raw_token_log_probs'] is None
            assert not context['records'][-1]['row']['done']
            assert assess(task,context,context['chosen'])[1] is None
            enc=encode_recovery_pair(tok,pair,100000);assert enc['chosen'][-1]==enc['rejected'][-1]==tok.eos_token_id
            tool=parse_action(pair['chosen'])['tool'];tools[tool]=tools.get(tool,0)+1;count+=1
    assert count==84 and tools=={'calculate_change':36,'query_sql':36,'check_result':12}

def test_recovery_after_unverified_sql_requeries_and_after_stale_ids_reuses_verified_ids():
    task=make_tasks()[0]
    for scenario in ('sql_unverified_requery','repeat_evidence_error'):
        c=build_context(task,scenario);assert parse_action(c['chosen'])['tool']=='query_sql'
        assert c['chosen_observation']['verified_metrics']==['current_net_cents']
        assert assess(task,c,c['scripted_rejected'])[0]['tool_error']=='provide verified monthly evidence IDs'
    c=build_context(task,'stale_evidence_repair')
    assert parse_action(c['chosen'])['tool']=='calculate_change'
    assert c['chosen_observation']['verified_metrics']==['change_cents']
    assert c['records'][-1]['row']['action']==c['records'][-2]['row']['action']

def test_native_context_sampling_and_scripted_provenance_are_separate():
    task=make_tasks()[0];tok=CharTokenizer();c=build_context(task,'format_repair')
    def sample(text):
        ids=[ord(ch)+1 for ch in text]+[0]
        return {'text':text,'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),'prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),'mode':'dpo_sample','finish_reason':'stop','sample_id':'native-0'}
    native=make_pair(task,c,tok,100000,sample('invalid JSON'));assert native['rejected_token_origin']=='native_sample'
    assert make_pair(task,c,tok,100000,sample(c['chosen'])) is None
    bad=sample('invalid JSON');bad['prompt_token_ids']=[1]
    with pytest.raises(ValueError,match='prompt'):make_pair(task,c,tok,100000,bad)
    bad=copy.deepcopy(native);bad['rejected_raw_token_log_probs']=None
    with pytest.raises(ValueError,match='alignment'):encode_recovery_pair(tok,bad,100000)
    synthetic=make_pair(task,c,tok,100000)
    bad=copy.deepcopy(synthetic);bad['rejected_token_origin']='native_sample'
    with pytest.raises(ValueError,match='provenance'):encode_recovery_pair(tok,bad,100000)
    assert len(select_pairs([], [synthetic], [native,native]))==2

def test_test_tasks_context_tamper_and_duplicate_context_caps():
    with pytest.raises(ValueError,match='train'):build_context(make_tasks()[-1],'format_repair')
    task=make_tasks()[0];tok=CharTokenizer();c=build_context(task,'repeat_sql_error');bad=copy.deepcopy(c);bad['records'][0]['row']['reward']+=1
    with pytest.raises(ValueError,match='replay'):assess(task,bad,bad['chosen'])
    pair=make_pair(task,c,tok,100000);assert len(select_pairs([pair],[pair],[]))==1


def test_recovery_worker_build_and_audit_regenerate_mixed_dataset(tmp_path,monkeypatch):
    import collect_v2_dpo_recovery as worker
    import transformers
    from dataclasses import asdict
    from apt_v2.dpo import pair_id
    from common import file_hash
    from apt_v2.sft import encode_example
    tasks=make_tasks()[:12];tm={t.id:t for t in tasks};tok=CharTokenizer()
    cfg={'seed':31,'samples_per_context':3,'pair_max_length':100000,'temperature':.8,'adapter_sha256':'pinned'}
    monkeypatch.setattr(worker,'verify_inputs',lambda output:(cfg,None))
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tok)
    contexts=[build_context(t,s) for t in tasks for s in SCENARIOS]
    scripted=[make_pair(tm[c['task_id']],c,tok,100000) for c in contexts]
    # Independent historical anchor fixtures use the original native-token pair schema.
    anchors=[]
    for i in range(44):
        c=contexts[i%len(contexts)];text='historical invalid '+str(i)
        enc=encode_example(tok,{'messages':c['messages'],'response':text},100000)
        p={'task_id':c['task_id'],'family_id':c['family_id'],'split':'train','messages':c['messages'],'chosen':c['chosen'],'rejected':text,
            'native_prompt_token_ids':enc['input_ids'][:enc['prompt_length']],
            'native_rejected_token_ids':enc['input_ids'][enc['prompt_length']:],'rejected_source':'frozen_sft_sample'}
        p['pair_id']=pair_id(p);anchors.append(p)
    samples=[]
    for c in contexts:
        for ri in range(3):
            text='{}' if ri!=1 else c['chosen'];ids=[ord(ch)+1 for ch in text]+[0]
            seed=31+list(tm).index(c['task_id'])*10000+list(SCENARIOS).index(c['scenario'])*100+ri
            samples.append({'text':text,'response_token_ids':ids,'raw_token_log_probs':[-.1]*len(ids),
                'prompt_token_ids':tok.apply_chat_template(c['messages'],add_generation_prompt=True),'mode':'dpo_sample','finish_reason':'stop',
                'context_id':c['context_id'],'task_id':c['task_id'],'scenario':c['scenario'],'sample_index':ri,
                'sample_seed':seed,'temperature':.8,'adapter_sha256':'pinned','sample_id':c['context_id']+':sample-'+str(ri)})
    for name,rows in [('contexts',contexts),('scripted_pairs',scripted),('anchor_pairs',anchors),('samples',samples)]:worker.jsonl(tmp_path/(name+'.jsonl'),rows)
    (tmp_path/'model_identity.json').write_text('{"files_sha256":{}}')
    (tmp_path/'collection_summary.json').write_text(json.dumps({'native_samples':252,'samples_sha256':file_hash(tmp_path/'samples.jsonl')}))
    worker.build(tmp_path);worker.audit(tmp_path)
    summary=json.loads((tmp_path/'pair_summary.json').read_text())
    assert summary['pairs']==212 and summary['native_failure_candidates']==168 and summary['selected_native_recovery_pairs']==84
    assert summary['pairs_by_rejected_source']=={'frozen_sft_sample':44,'scripted_failure':84,'frozen_dpo_sample':84}
    pairs=worker.load(tmp_path/'pairs.jsonl');assert pairs[:44]==anchors
    samples[0]['sample_seed']+=1;(tmp_path/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    (tmp_path/'collection_summary.json').write_text(json.dumps({'native_samples':252,'samples_sha256':file_hash(tmp_path/'samples.jsonl')}))
    with pytest.raises(ValueError,match='identity'):worker.derive(tmp_path)

def test_recovery_runner_failure_retains_logs_and_frozen_status(tmp_path,monkeypatch):
    import run_v2_dpo_recovery as runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    (tmp_path/'configs').mkdir();(tmp_path/'scripts').mkdir();candidate=tmp_path/'artifacts/candidate'
    for name in ('pairs.jsonl','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        p=candidate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture')
    (tmp_path/'configs/dpo-recovery-collect-v1.json').write_text(json.dumps({'run_id':'recovery','source_run':'candidate','max_wall_seconds':60}))
    (tmp_path/'dpo_recovery_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'scripts/collect_v2_dpo_recovery.py').write_text('raise RuntimeError("intentional")')
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/recovery/run_status.json').read_text())
    assert status['optimizer_updates']==0 and status['training_enabled'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'recovery-review.zip') as z:assert 'recovery/prepare.log' in z.namelist()
