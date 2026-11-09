"""Diagnostic contracts: fixed contexts, causal scores and failure evidence."""
import sys,json,copy
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from apt_v2.sft import build_dataset,action_loss
from apt_v2.sft_diagnostics import select_probes,grouped_scores,compare_scores,score_action,summarize_episodes

@pytest.fixture(scope='module')
def dataset():return build_dataset()

def test_probes_include_actual_recovery_contexts(dataset):
    probes=select_probes(dataset[1]);assert len(probes)==8
    assert len({p['probe_id'] for p in probes})==8
    assert {json.loads(p['response'])['tool'] for p in probes[:6]}=={
        'read_policy','inspect_schema','query_sql','calculate_change','check_result','submit'}
    assert 'no such table: metrics' in probes[6]['messages'][-1]['content']
    assert 'provide verified monthly evidence IDs' in probes[7]['messages'][-1]['content']
    assert probes[6]['variant']=='sql_repair' and probes[7]['variant']=='evidence_repair'
    assert all(p['split']=='train' for p in probes)

def test_grouped_scores_keep_token_and_action_means_separate():
    rows=[{'tool':'query_sql','variant':'expert','tokens':2,'nll_sum':2.,'correct_tokens':1,'all_tokens_correct':False},
        {'tool':'submit','variant':'expert','tokens':6,'nll_sum':12.,'correct_tokens':6,'all_tokens_correct':True}]
    summary=grouped_scores(rows)['all']
    assert summary['token_mean_nll']==1.75 and summary['action_mean_nll']==1.5
    assert summary['token_accuracy']==7/8 and summary['teacher_forced_exact_actions']==1

def test_compare_scores_checks_shapes_and_argmax():
    a={'token_log_probs':[-.2,-.5],'argmax_token_ids':[2,3]}
    b={'token_log_probs':[-.3,-.5],'argmax_token_ids':[2,4]}
    d=compare_scores(a,b)
    assert d['argmax_disagreements']==1 and d['max_abs_logp_difference']==pytest.approx(.1)
    with pytest.raises(ValueError,match='unaligned'):compare_scores(a,{**b,'argmax_token_ids':[2]})
    with pytest.raises(ValueError,match='finite'):compare_scores(a,{**b,'token_log_probs':[float('nan'),-.5]})

def test_real_qwen_diagnostic_score_matches_training_loss():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from peft import get_peft_model,LoraConfig
    torch.manual_seed(17)
    model=get_peft_model(Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64)),
        LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
    model.requires_grad_(False)
    sample={'input_ids':[1,4,5,6,2],'labels':[-100,-100,-100,6,2],'prompt_length':3,'response_length':2}
    actual=score_action(model,sample)
    assert actual['nll_sum']/actual['tokens']==pytest.approx(action_loss(model,sample).item(),abs=1e-6)
    assert len(actual['token_log_probs'])==len(actual['argmax_token_ids'])==2
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    with torch.no_grad():
        for n,p in model.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:p.fill_(.2)
    enabled=score_action(model,sample)
    with model.disable_adapter():disabled=score_action(model,sample)
    assert compare_scores(enabled,disabled)['max_abs_logp_difference']>0

def test_successful_episode_summary(dataset):
    episodes=[e for e in dataset[0] if e['variant']=='expert']
    summary=summarize_episodes(episodes)
    assert summary['tasks']==12 and summary['successes']==12
    assert summary['fault_tasks']==6 and summary['recovered']==6
    assert summary['valid_sql']==24 and summary['verified_metric_events']==36

def test_diagnostic_runner_failure_retains_review(tmp_path,monkeypatch):
    import run_v2_sft_diagnostics as runner
    (tmp_path/'configs').mkdir();(tmp_path/'diagnostics').mkdir()
    pilot=tmp_path/'artifacts/pilot';(pilot/'final/adapter').mkdir(parents=True)
    for name in ('examples.jsonl','data_summary.json','model_identity.json','run_manifest.json'):
        (pilot/name).write_text('{}')
    (pilot/'final/complete.json').write_text('{"adapter_sha256":{}}')
    (pilot/'final/adapter/adapter_model.safetensors').write_bytes(b'test')
    (tmp_path/'configs/sft-diagnostics-v1.json').write_text(json.dumps({'run_id':'diagnostic','pilot_run':'pilot','max_wall_seconds':60}))
    (tmp_path/'sft_diagnostics_release_manifest.json').write_text('{"files":{}}')
    (tmp_path/'diagnostics/diagnose_sft_v1.py').write_text('raise RuntimeError("intentional stage failure")')
    original=runner.file_hash
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'source_files',lambda:{})
    monkeypatch.setattr(runner,'file_hash',lambda p:'95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e'
        if p.name=='adapter_model.safetensors' else original(p))
    with pytest.raises(RuntimeError,match='prepare failed'):runner.run()
    status=json.loads((tmp_path/'artifacts/diagnostic/run_status.json').read_text())
    assert status['stages']==[{'stage':'prepare','exit_code':1}]
    assert status['optimizer_updates']==0 and status['training_enabled'] is False
    import zipfile
    with zipfile.ZipFile(tmp_path/'diagnostic-review.zip') as z:
        assert 'diagnostic/prepare.log' in z.namelist()
    with pytest.raises(FileExistsError):runner.run()


def test_probe_record_preserves_branch_separate_from_sampling_mode():
    from apt_v2.runtime import Generation
    from apt_v2.sft_diagnostics import probe_record
    sample=Generation('action',mode='greedy',prompt_token_ids=[1],response_token_ids=[2])
    rows=[probe_record('probe-00',mode,sample,{'token_log_probs':[-.2]}) for mode in ('base','sft')]
    assert {r['mode'] for r in rows}=={'base','sft'}
    assert all(r['sampling_mode']=='greedy' for r in rows)
    assert len({(r['mode'],r['probe_id']) for r in rows})==2


def test_probe_index_rejects_legacy_collisions_and_missing_rows():
    from apt_v2.sft_diagnostics import index_probes
    rows=[{'mode':'base','probe_id':'p'},{'mode':'sft','probe_id':'p'}]
    assert len(index_probes(rows,['p']))==2
    with pytest.raises(ValueError,match='duplicate'):index_probes(rows+[rows[0]],['p'])
    with pytest.raises(ValueError,match='unexpected'):index_probes([{'mode':'greedy','probe_id':'p'}],['p'])
    with pytest.raises(ValueError,match='missing'):index_probes(rows[:1],['p'])
