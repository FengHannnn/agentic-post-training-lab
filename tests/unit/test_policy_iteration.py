import copy
import json
from pathlib import Path
import sys

import pytest

from agentic_post_training.trainers.policy_iteration import (
    digest_json, verify_files, file_hash, seed_for, select_mixed_batches)


def test_new_seed_range_is_unique_and_disjoint_from_old_rollout():
    new = {seed_for(g, i) for g in range(144) for i in range(8)}
    old = {20261021 + g * 1000 + i for g in range(144) for i in range(8)}
    assert len(new) == 1152 and not new & old
    with pytest.raises(ValueError):
        seed_for(144, 0)


def test_all_mixed_groups_once_in_frozen_order_and_no_signal():
    def group(key, values):
        return {'group_id': key, 'rewards': values, 'eligible': True, 'zero_variance': len(set(values)) == 1}
    rewards = {'groups': [group('b', [0, 1] * 4), group('a', [1] * 8), group('c', [0, 1] * 4)]}
    assert select_mixed_batches(rewards, ['c', 'a', 'b']) == [['c', 'b']]
    assert select_mixed_batches({'groups': [group('a', [0] * 8)]}, ['a']) == []
    rewards['groups'][0]['zero_variance'] = True
    with pytest.raises(ValueError, match='variance'):
        select_mixed_batches(rewards, ['a', 'b', 'c'])


def test_binding_changes_when_parent_rollout_reference_or_plan_changes():
    original = {'parent': 'p1', 'rollout': 'r1', 'reference': 'sft', 'plan': {'lr': 1e-5}}
    bound = digest_json(original)
    assert bound == digest_json(dict(reversed(list(original.items()))))
    for key in original:
        changed = copy.deepcopy(original)
        changed[key] = 'different'
        assert digest_json(changed) != bound


def test_manifest_rejects_outside_and_changed_files(tmp_path):
    folder = tmp_path / 'run'
    folder.mkdir()
    path = folder / 'sample'
    path.write_bytes(b'original')
    digest = file_hash(path)
    verify_files(folder, {'sample': digest})
    outside = tmp_path / 'outside'
    outside.write_bytes(b'original')
    with pytest.raises(ValueError, match='unsafe'):
        verify_files(folder, {'../outside': digest})
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_files(folder, {'sample': digest})


def test_new_native_records_reject_old_seed_and_reward_corruption():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts'))
    from iteration_common import validate_records
    from agentic_post_training.evaluation.tool_calls import EvalCase
    from agentic_post_training.rollouts.records import RolloutSample
    from agentic_post_training.rewards.contract import score_reward
    from dataclasses import asdict

    text = '{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}'
    class Tokenizer:
        eos_token_id = pad_token_id = 0
        def decode(self, ids, **kwargs):
            return text if kwargs['skip_special_tokens'] else text + '<EOS>'
    case = EvalCase('case-natural-right_grouped', 'math', '1+2', 'tool_call', '1+2', 3)
    rows = []
    for i in range(8):
        sample = RolloutSample(case.id, case.id, i, seed_for(0, i), [1], [5, 0], [-.1, -.1],
                               [-.1, -.1], 0, 'eos', text, text + '<EOS>')
        rw = score_reward(text, case, finish_reason='eos')
        rows.append({'case': asdict(case), 'sample': asdict(sample), 'score': rw['score'],
                     'semantic_pass': rw['semantic_pass']})
    _, batch, reward = validate_records(rows, [case], [[1]], Tokenizer())
    assert batch['response_mask'].sum() == 16 and reward['reward_sum'] == 8
    corrupt = copy.deepcopy(rows)
    corrupt[0]['sample']['seed'] = 20261021
    with pytest.raises(ValueError, match='identity'):
        validate_records(corrupt, [case], [[1]], Tokenizer())
    rows[0]['score']['end_to_end_pass'] = False
    with pytest.raises(ValueError):
        validate_records(rows, [case], [[1]], Tokenizer())


def test_three_way_comparison_exposes_equal_total_regression_and_length():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts'))
    from train_grpo_iteration import compare_three
    def row(key, passed):
        return {'case_id': key, 'text': str(passed), 'strict_pass': passed, 'semantic_pass': passed,
                'finish_reason': 'eos', 'score': dict.fromkeys(('json_valid', 'schema_valid', 'route_correct'), passed)}
    sft = {'development': [row('a', True), row('b', False)]}
    current = {'development': [row('a', False), row('b', True)]}
    result = compare_three(sft, copy.deepcopy(sft), current)['previous_to_current']['development']
    assert result['before_strict'] == result['after_strict'] == 1
    assert result['strict_improved_case_ids'] == ['b'] and result['strict_regressed_case_ids'] == ['a']
    assert result['new_structure_failure_cases'] == 1
    current['development'][1]['finish_reason'] = 'length'
    result = compare_three(sft, sft, current)['sft_to_current']['development']
    assert result['after_strict'] == 0


@pytest.mark.parametrize('refresh_code,expected_calls', [(0, 2), (2, 1)])
def test_driver_stops_on_refresh_failure_and_always_packages_evidence(tmp_path, monkeypatch, refresh_code, expected_calls):
    from types import SimpleNamespace
    import zipfile
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'scripts'))
    import run_grpo_iteration as driver
    (tmp_path / 'configs').mkdir()
    plan = tmp_path / 'configs/plan.json'
    plan.write_text(json.dumps({'source_rollout': 'artifacts/trajectories/new'}), encoding='utf-8')
    stub = tmp_path / 'stub'
    stub.write_text('verified', encoding='utf-8')
    manifest = tmp_path / 'bundle.json'
    manifest.write_text(json.dumps({'files': {'stub': file_hash(stub)},
                                    'required_existing_files': {'stub': file_hash(stub)}}), encoding='utf-8')
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'PLAN_PATH', plan)
    monkeypatch.setattr(driver, 'BUNDLE_MANIFEST', manifest)
    monkeypatch.setattr(driver, 'OUTPUT', tmp_path / 'artifacts/checkpoints/new')
    monkeypatch.setattr(driver, 'DRIVER', tmp_path / 'artifacts/evaluations/driver')
    monkeypatch.setattr(driver, 'ARCHIVE', tmp_path / 'review.zip')
    monkeypatch.setattr(driver, 'preflight', lambda: {'input_preflight_passed': True})
    calls = []
    def invoke(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=refresh_code if len(calls) == 1 else 0)
    if refresh_code:
        with pytest.raises(RuntimeError, match='refresh failed'):
            driver.dispatch(2.18, invoke=invoke)
    else:
        driver.dispatch(2.18, invoke=invoke)
    assert len(calls) == expected_calls
    with zipfile.ZipFile(driver.ARCHIVE) as z:
        result = json.loads(z.read('evaluations/driver/dispatch.json'))
        assert len(result['stages']) == expected_calls and not result['automatic_retry']
    with pytest.raises(FileExistsError):
        driver.dispatch(2.18, invoke=invoke)
    assert len(calls) == expected_calls
