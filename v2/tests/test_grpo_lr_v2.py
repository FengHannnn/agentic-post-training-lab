"""Contracts for changing LR while retaining the exact r16 supervision."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from grpo_common import atomic_json, read
from apt_v2 import grpo_dose_study as dose, grpo_lr_study as lr
from test_grpo_repair_study_v2 import result


def master():
    return read(ROOT/'configs/grpo-sql-lr-study-v1.json')


@pytest.mark.parametrize('name,rate', lr.CANDIDATES.items())
def test_recipe_and_order_match_r16_except_learning_rate(name, rate):
    from grpo_repair_worker import sft_order
    cfg = master()
    warm = lr.warm_config(cfg, rate)
    assert warm['run_id'] == f'{lr.RUN_ID}-{name}'
    assert warm['sft'] == dict(dose.recipe(16), learning_rate=rate)
    expected = read(ROOT/'artifacts/inputs/grpo-sql-dose-20261110/order-r16.json')
    assert sft_order(warm) == expected
    with pytest.raises(ValueError, match='exact published r16'):
        lr.order(warm, read(ROOT/cfg['sft_examples']), expected[::-1])
    warm['sft']['seed'] += 1
    with pytest.raises(ValueError, match='warm recipe'):
        lr.validate_warm(warm)


@pytest.mark.parametrize('change', [
    {'learning_rates': [5e-6, 3e-6]}, {'max_wall_seconds': 21600}, {'target_repeats': 32},
    {'prior_elapsed_seconds': 0}, {'run_grpo': True},
])
def test_reject_unregistered_experiment_changes(change):
    cfg = master()
    lr.validate_master(cfg)
    evidence = read(ROOT/cfg['input_dir']/'prior-evidence.json')
    lr.validate_prior(evidence, cfg)
    with pytest.raises(ValueError, match='contract'):
        lr.validate_master({**cfg, **change})
    evidence['receipts'][-1]['elapsed_seconds_including_packaging'] -= 1
    with pytest.raises(ValueError, match='ledger'):
        lr.validate_prior(evidence, cfg)


@pytest.mark.parametrize('mode,selected', [
    ('first_pass', 'lr3'), ('train_skip', 'lr5'), ('full_reject', 'lr5'), ('both_reject', None),
])
def test_actual_runner_uses_ordered_gates_and_stops_on_first_ready_b(tmp_path, monkeypatch, mode, selected):
    import grpo_dose_worker as worker
    import run_v2_grpo_dose as runner
    cfg = master()
    study = runner.Study(cfg, tmp_path)
    calls = []

    def child(name):
        path = tmp_path/'branches'/name
        atomic_json(path/'config.json', read(ROOT/cfg['branch_template']))
        return path

    def stage(label, *args):
        calls.append(label)
        if label.startswith('gate-'):
            name = label.removeprefix('gate-')
            passed = mode != 'both_reject' and (mode != 'train_skip' or name != 'lr3')
            atomic_json(tmp_path/'train-gates'/(name+'.json'), {'passed': passed})

    def evaluate(child, name, *args):
        if name == 'w':
            return dict(result(first=0), elapsed_seconds=1800)
        calls.append('evaluate-'+name)
        return result(full=13 if mode == 'full_reject' and name == 'lr3' else 14, target=8, first=9)

    monkeypatch.setattr(study, 'child', child)
    monkeypatch.setattr(study, 'stage', stage)
    monkeypatch.setattr(study, 'evaluate', evaluate)
    monkeypatch.setattr(worker, 'candidate_source', lambda *a: {'adapter_sha256': 'a'*64, 'source_sft_updates': 573})
    decision = study.execute()
    assert decision['selected_warm_start'] == selected
    assert decision['formal_baseline'] == 'W' and decision['run_grpo'] is False
    assert [s for s in calls if s.startswith('warm-')] == (['warm-lr3'] if selected == 'lr3' else ['warm-lr3', 'warm-lr5'])
    if mode in ('train_skip', 'both_reject'):
        assert 'evaluate-lr3' not in calls
    ready = tmp_path/'ready-B.json'
    if selected:
        metadata = read(ready)
        assert metadata['learning_rate'] == lr.CANDIDATES[selected]
        assert metadata['source_sft_updates'] == 573 and metadata['source_rl_updates'] == 0
    else:
        assert not ready.exists()


def test_branch_names_bind_candidate_lr_and_ancestry(tmp_path, monkeypatch):
    import run_v2_grpo_dose as runner
    from grpo_common import validate_config
    cfg = master()
    atomic_json(tmp_path/cfg['branch_template'], read(ROOT/cfg['branch_template']))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    out = tmp_path/'artifacts'/lr.RUN_ID
    assert runner.branch_config(cfg, out, 'w')['source_sft_updates'] == 513
    for name, rate in lr.CANDIDATES.items():
        adapter = out/'warmups'/name/'warm-start/B/adapter/adapter_model.safetensors'
        adapter.parent.mkdir(parents=True)
        adapter.write_bytes(name.encode())
        branch = runner.branch_config(cfg, out, name)
        assert branch['source_sft_updates'] == 573 and branch['warm_learning_rate'] == rate
        with pytest.raises(ValueError, match='candidate identity'):
            validate_config({**branch, 'warm_learning_rate': next(r for r in lr.CANDIDATES.values() if r != rate)})
