import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import explore_grpo_sampling as probe
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.evaluation.tool_calls import EvalCase


PLAN = {'temperatures': {'t100': 1., 't130': 1.3, 't160': 1.6}, 'sample_prefixes': [8, 16, 32]}
CASE = EvalCase('case', 'math', 'calculate', 'tool_call', '1+2', 3)
ANSWER = '{"type":"tool_call","name":"calculator","arguments":{"expression":"1+2"}}'


class Tokenizer:
    eos_token_id = 0

    def decode(self, tokens, *, skip_special_tokens, clean_up_tokenization_spaces):
        value = ANSWER if tokens[0] == 1 else 'bad json'
        return value + ('<eos>' if not skip_special_tokens and tokens[-1] == 0 else '')


def record(arm='t100', index=0, good=True, length=False):
    tokens = ([1 if good else 2] * 64) if length else [1 if good else 2, 0]
    native = SimpleNamespace(prompt_token_ids=[9], outputs=[SimpleNamespace(
        token_ids=tokens, logprobs=[{t: SimpleNamespace(logprob=-.7)} for t in tokens],
        finish_reason='length' if length else 'stop')])
    return probe.native_record(native, [9], CASE, arm, index, probe.seed_for(0, index),
                               PLAN['temperatures'][arm], Tokenizer())


def full_rows():
    rows = []
    for arm, group, index, seed in probe.requests():
        row = record(arm, index)
        row['seed'] = seed
        rows.append(row)
    return rows


def test_schedule_is_complete_balanced_and_paired_without_reusing_historical_seeds():
    requests = probe.requests()
    assert len(requests) == len(set(requests)) == 2304
    for arm in PLAN['temperatures']:
        part = [(g, i, seed) for a, g, i, seed in requests if a == arm]
        assert len(part) == 768
        assert {(g, i) for g, i, seed in part} == {(g, i) for g in range(24) for i in range(32)}
        assert len({seed for g, i, seed in part}) == 768
        assert all(seed == probe.seed_for(g, i) for g, i, seed in part)
    assert [requests[block * 576][0] for block in range(4)] == ['t100', 't130', 't160', 't100']
    old = {base + g * 1000 + i for base in (20261021, 202610280) for g in range(144) for i in range(8)}
    assert not {r[3] for r in requests} & old


def test_selection_uses_only_training_and_rejects_duplicate_historical_groups():
    folder = probe.ROOT / 'data/preferences/dpo_v2'
    splits = load_preference_splits(folder, probe.read(folder / 'manifest.json'))
    arithmetic = [r for r in splits['train'] if r.expected_type == 'tool_call']
    audit = {'groups': [{'group_id': r.id, 'eligible': True, 'rewards': [0.] * 8} for r in arithmetic]}
    selected, families = probe.select_cases(splits['train'], audit, audit)
    assert len(selected) == 24 and len(families) == 4
    assert not {r.id for r in selected} & {r.id for r in splits['development']}
    assert all(sum(r.family_id == family for r in selected) == 6 for family in families)
    broken = copy.deepcopy(audit)
    broken['groups'].append(broken['groups'][0])
    with pytest.raises(ValueError, match='cover all'):
        probe.select_cases(splits['train'], broken, audit)


def test_raw_logp_is_never_admitted_as_hotter_behavior_probability():
    cold, hot = record(), record('t130')
    assert cold['reward'] == hot['reward'] == 1.
    assert cold['behavior_token_log_probs'] == cold['sampler_raw_model_token_log_probs']
    assert hot['behavior_token_log_probs'] is None
    assert not hot['training_admitted']
    truncated = record(length=True)
    assert truncated['reward'] == 0. and truncated['semantic_pass']
    assert truncated['finish_reason'] == 'length' and len(truncated['response_mask']) == 64


@pytest.mark.parametrize('field,value', [
    ('seed', -1), ('response_mask', [True, False]), ('temperature', 1.6),
    ('sampler_raw_model_token_log_probs', [float('nan'), -.7]),
    ('reward', 0.), ('finish_reason', 'length'), ('training_admitted', True),
])
def test_full_record_audit_rejects_identity_probability_mask_and_reward_corruption(field, value):
    rows = full_rows()
    probe.validate_records(rows, PLAN, [CASE] * 24, [[9]] * 24, Tokenizer())
    rows[0][field] = value
    with pytest.raises(ValueError):
        probe.validate_records(rows, PLAN, [CASE] * 24, [[9]] * 24, Tokenizer())


def test_full_audit_rejects_fabricated_hot_behavior_and_missing_records():
    rows = full_rows()
    next(r for r in rows if r['arm'] == 't130')['behavior_token_log_probs'] = [-.7, -.7]
    with pytest.raises(ValueError, match='semantics'):
        probe.validate_records(rows, PLAN, [CASE] * 24, [[9]] * 24, Tokenizer())
    with pytest.raises(ValueError, match='2304'):
        probe.validate_records(rows[:-1], PLAN, [CASE] * 24, [[9]] * 24, Tokenizer())


def test_summary_separates_nested_count_gain_equal_budget_temperature_and_format_cost():
    selected = [SimpleNamespace(id=f'case-{g}', family_id=f'family-{g//6}',
                provenance={'wording': probe.STRATA[g % 6][0]}, category=probe.STRATA[g % 6][1])
                for g in range(24)]
    rows = []
    for arm, group, index, seed in probe.requests():
        # At T=1 group 0 first succeeds at sample 10. Hotter arm loses it but reaches group 1.
        good = (arm == 't100' and group == 0 and index == 9) or (arm == 't130' and group == 1 and index == 0)
        row = record(arm, index, good=good)
        row['case_id'] = selected[group].id
        rows.append(row)
    summary = probe.summarize(rows, selected, PLAN)
    base = summary['by_arm_and_prefix']['t100']
    assert base['8']['overall']['strict_successes'] == 0
    assert base['16']['overall']['strict_successes'] == base['32']['overall']['strict_successes'] == 1
    assert base['32']['by_role']['persistent_zero']['mixed_groups'] == 1
    assert base['32']['by_role']['persistent_zero']['nonzero_advantage_completions_if_grouped'] == 32
    assert base['32']['per_case'][0]['first_strict_success_at'] == 10
    assert summary['count_effect_t100_prefix32_vs8']['primary_newly_reached_case_ids'] == ['case-0']
    comparison = summary['temperature_equal_budget_32']['t130']
    assert comparison['primary_newly_reached_case_ids'] == ['case-1']
    assert comparison['primary_lost_case_ids'] == ['case-0']
    assert comparison['overall_invalid_json_delta'] == 0
    assert base['32']['overall']['strict_success_per_1000_samples'] == pytest.approx(1000/768)
    with pytest.raises(ValueError, match='duplicate'):
        probe.summarize(rows[:-1] + [rows[0]], selected, PLAN)


def test_failed_model_stage_stops_before_generation_and_packs_evidence(tmp_path, monkeypatch):
    output = tmp_path / 'failed-run'
    archive = tmp_path / 'review.zip'
    monkeypatch.setattr(probe, 'ARCHIVE', archive)
    monkeypatch.setattr(probe, 'read', lambda p: {'files': {}, 'required_existing_files': {}})
    monkeypatch.setattr(probe, 'verify_files', lambda *a: None)
    monkeypatch.setattr(probe, 'preflight', lambda: {})
    calls = []

    class FailedProcess:
        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self, timeout):
            return 1

    monkeypatch.setattr(probe.subprocess, 'Popen', FailedProcess)
    with pytest.raises(RuntimeError, match='model-check exited 1'):
        probe.run(SimpleNamespace(hourly_price=2.18, output=output))
    assert len(calls) == 1 and calls[0][3] == 'model-check'
    cost = json.loads((output / 'cost_and_run.json').read_text())
    assert not cost['automatic_retry'] and cost['stages'] == [{'stage': 'model-check', 'exit_code': 1}]
    with probe.zipfile.ZipFile(archive) as z:
        assert z.namelist() == ['failed-run/cost_and_run.json']
