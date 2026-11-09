"""Exercise the real new-iteration audit on native records, without GPU scores."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import no_update_metrics, pack_trainer_scores
from agentic_post_training.trainers.policy_iteration import file_hash, seed_for


def test_fresh_batch_audit_recomputes_identity_rewards_scores_and_summary(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    import collect_grpo_iteration as collector
    from iteration_common import validate_records

    text = '{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}'
    class Tokenizer:
        eos_token_id = pad_token_id = 0
        def decode(self, tokens, **kwargs):
            value = text if tokens[0] == 5 else 'bad'
            return value if kwargs['skip_special_tokens'] else value + '<EOS>'
    tok = Tokenizer()
    cases = [EvalCase(f'case-{i}-natural-right_grouped', 'math', '1+2', 'tool_call', '1+2', 3) for i in range(144)]
    prompts = [[1] for _ in cases]
    plan = {'contract_version': 'grpo-policy-iteration-v1', 'parent_parameter_sha256': 'candidate'}
    inv = {'source_sha256': {'stub': file_hash(Path(__file__))},
           'policy_version': {'sampler_and_old': 'candidate', 'reference': 'sft'}}
    monkeypatch.setattr(collector, 'ROOT', tmp_path)
    monkeypatch.setattr(collector, 'setup', lambda: ({}, plan, tmp_path, tmp_path, tok, cases, prompts))
    monkeypatch.setattr(collector, 'identity', lambda *args: inv)
    (tmp_path / 'stub').write_bytes(Path(__file__).read_bytes())
    output = tmp_path / 'output'
    (output / 'source').mkdir(parents=True)
    (output / 'source/stub').write_bytes(Path(__file__).read_bytes())
    def write(name, value):
        (output / name).write_text(json.dumps(value), encoding='utf-8')
    write('run_manifest.json', {**inv, 'device': 'NVIDIA L20'})
    rows = []
    for g, case in enumerate(cases):
        for i in range(8):
            tokens = [5 if i < 4 else 6, 0]
            sample = RolloutSample(case.id, case.id, i, seed_for(g, i), [1], tokens,
                [-.1, -.1], [-.1, -.1], 0, 'eos', tok.decode(tokens, skip_special_tokens=True),
                tok.decode(tokens, skip_special_tokens=False))
            rw = score_reward(sample.text, case, finish_reason='eos')
            rows.append({'case': asdict(case), 'sample': asdict(sample), 'score': rw['score'],
                         'semantic_pass': rw['semantic_pass']})
    (output / 'rollouts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    groups, _, rewards = validate_records(rows, cases, prompts, tok)
    reward_map = {(r['group_id'], r['sample_index']): r for r in rewards['records']}
    records = []
    for group in groups:
        for s in group:
            old = torch.tensor([-.1, -.1], dtype=torch.float32)
            new = old.clone().requires_grad_(True)
            rw = reward_map[s.group_id, s.sample_index]
            records.append({'group_id': s.group_id, 'sample_index': s.sample_index,
                'response_token_ids': s.response_token_ids, 'sampler_behavior_token_log_probs': s.behavior_token_log_probs,
                'trainer_old_token_log_probs': old.tolist(), 'trainer_new_token_log_probs': new.detach().tolist(),
                'no_update_check': no_update_metrics(old, new, 1e-5),
                **{k: rw[k] for k in ('eligible', 'reward', 'advantage')}})
    (output / 'trainer_scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    torch.save(pack_trainer_scores(groups, records, 0), output / 'trainer_score_batch.pt')
    write('reward_audit.json', rewards)
    write('generation_checks.json', {'effective_engine': collector.EFFECTIVE, 'sampling_parameters': collector.sampling(0),
        'generation_requests': 1152, 'engine_generate_calls': 18, 'chunk_requests': 64,
        'generation_seconds': 1., 'response_tokens': 2304, 'engine_shutdown_called': True})
    write('trainer_checks.json', {'initial_parameter_sha256': 'candidate', 'final_parameter_sha256': 'candidate',
        'no_grad_accumulated': True, 'model_backward_calls': 0, 'optimizer_steps': 0, 'scoring_seconds': 1.})
    correction = Path(__file__).resolve().parents[2] / 'configs/rollout_correction_contract.json'
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/rollout_correction_contract.json').write_bytes(correction.read_bytes())
    result = collector.audit(output)
    assert result['signal_collection_interface_passed'] and result['full8_mixed_groups'] == 144
    assert result['policy_version']['sampler_and_old'] != result['policy_version']['reference']
    assert collector.audit(output, write_summary=False) == result
    result['reward_sum'] += 1
    write('summary.json', result)
    with pytest.raises(ValueError, match='summary differs'):
        collector.audit(output, write_summary=False)
    result['reward_sum'] -= 1
    write('summary.json', result)
    records[0]['trainer_old_token_log_probs'][0] = -.5
    (output / 'trainer_scores.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
    with pytest.raises(ValueError, match='no-update'):
        collector.audit(output, write_summary=False)
