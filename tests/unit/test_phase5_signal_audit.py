"""Synthetic 1152-record CPU integration audit; no model forward or GPU."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from agentic_post_training.rollouts.phase5_signal import sample_from_output,validate_records,sample_seed
from agentic_post_training.rollouts.trainer_scores import no_update_metrics,pack_trainer_scores
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward
from tests.unit.test_phase5_signal import Tokenizer,native


def test_full_1152_record_auditor_checks_masks_advantages_and_signal_counts(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('phase5_audit_test',root/'scripts/collect_phase5_signal.py')
    script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
    tok=Tokenizer();cases=[EvalCase(f'{g}-natural-right_grouped','math','1+2','tool_call','1+2',3) for g in range(144)]
    prompts=[[1] for _ in cases];rows=[]
    for g,case in enumerate(cases):
        for i in range(8):
            good=g%3==0 or (g%3==2 and i>=4)
            s=sample_from_output(native([5 if good else 6,0]),[1],case,i,sample_seed(g,i),tok)
            rw=score_reward(s.text,case,finish_reason='eos')
            rows.append({'sample':asdict(s),'case':asdict(case),'score':rw['score'],'semantic_pass':rw['semantic_pass']})
    groups,packed,rewards=validate_records(rows,cases,prompts,tok)
    indexed={(r['group_id'],r['sample_index']):r for r in rewards['records']}
    records=[]
    for group in groups:
        for s in group:
            old=torch.tensor(s.policy_token_log_probs,dtype=torch.float32)
            records.append({'group_id':s.group_id,'sample_index':s.sample_index,'response_token_ids':s.response_token_ids,
                'sampler_behavior_token_log_probs':s.behavior_token_log_probs,'trainer_old_token_log_probs':old.tolist(),
                'trainer_new_token_log_probs':old.tolist(),'no_update_check':no_update_metrics(old,old.clone().requires_grad_(),1e-5),
                **{k:indexed[s.group_id,s.sample_index][k] for k in ('eligible','reward','advantage')}})
    monkeypatch.setattr(script,'load_run',lambda out:({}, {'contract_version':'phase5-signal-144x8-v1'},None,None,tok,rows,groups,packed,rewards))
    def write(name,data):(tmp_path/name).write_text(json.dumps(data),encoding='utf-8')
    for name,data in [('rollouts.jsonl',rows),('trainer_scores.jsonl',records)]:
        (tmp_path/name).write_text('\n'.join(json.dumps(r) for r in data),encoding='utf-8')
    write('reward_audit.json',rewards);write('run_manifest.json',{'synthetic_test':True})
    digest=json.loads((root/'configs/trainer_probability_contract.json').read_text())['frozen_fp32_parameter_sha256']
    write('trainer_checks.json',{'initial_parameter_sha256':digest,'final_parameter_sha256':digest,
        'no_grad_accumulated':True,'model_backward_calls':0,'optimizer_steps':0,'scoring_seconds':1.})
    write('generation_checks.json',{'effective_engine':script.EFFECTIVE,'sampling_parameters':script.sampling(0),
        'generation_requests':1152,'engine_generate_calls':18,'chunk_requests':64,'generation_seconds':1.,
        'response_tokens':2304,'engine_shutdown_called':True})
    batch=pack_trainer_scores(groups,records,0);torch.save(batch,tmp_path/'trainer_score_batch.pt')
    script.audit(tmp_path)
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['signal_collection_interface_passed'] and not result['training_admitted']
    assert result['group_outcomes']=={'all_correct':48,'all_wrong':48,'mixed':48}
    assert result['prefix4_mixed_groups']==0 and result['extra_mixed_groups_from_samples_5_to_8']==48
    batch['response_mask'][0,0,0]=False;torch.save(batch,tmp_path/'trainer_score_batch.pt')
    with pytest.raises(ValueError,match='saved scores/masks differ'):script.audit(tmp_path)
