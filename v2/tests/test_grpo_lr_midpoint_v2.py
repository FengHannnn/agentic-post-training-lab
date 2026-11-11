"""Exercise the midpoint contract, runner, and independently checked diagnostics."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from apt_v2 import grpo_dose_study as dose, grpo_lr_midpoint as midpoint
from grpo_common import atomic_json, read, validate_config
from test_grpo_repair_study_v2 import result


def master():
    return read(ROOT/'configs/grpo-sql-lr-midpoint-v1.json')


def test_midpoint_retains_the_exact_r16_recipe_and_budget():
    from grpo_repair_worker import sft_order
    cfg = master()
    midpoint.validate_master(cfg)
    prior = read(ROOT/cfg['input_dir']/'prior-evidence.json')
    midpoint.validate_prior(prior, cfg)
    warm = midpoint.warm_config(cfg, 4e-6)
    assert warm['sft'] == dict(dose.recipe(16), learning_rate=4e-6)
    assert sft_order(warm) == read(ROOT/'artifacts/inputs/grpo-sql-lr-20261111/order-r16.json')
    assert cfg['max_wall_seconds'] == 13500
    assert cfg['prior_elapsed_seconds'] + cfg['max_wall_seconds'] < cfg['total_authorized_seconds']
    with pytest.raises(ValueError, match='4e-6'):
        midpoint.warm_config(cfg, 5e-6)
    with pytest.raises(ValueError, match='frozen midpoint'):
        midpoint.validate_master({**cfg, 'max_wall_seconds': 13685})
    prior['receipts'][-1]['elapsed_seconds_including_packaging'] -= 1
    with pytest.raises(ValueError, match='ledger'):
        midpoint.validate_prior(prior, cfg)


def test_branch_binds_w_or_lr4_with_expected_ancestry(tmp_path, monkeypatch):
    import run_v2_grpo_dose as runner
    cfg = master()
    atomic_json(tmp_path/cfg['branch_template'], read(ROOT/cfg['branch_template']))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    output = tmp_path/'artifacts'/cfg['run_id']
    assert runner.branch_config(cfg, output, 'w')['source_sft_updates'] == 513
    adapter = output/'warmups/lr4/warm-start/B/adapter/adapter_model.safetensors'
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b'lr4')
    branch = runner.branch_config(cfg, output, 'lr4')
    assert branch['warm_learning_rate'] == 4e-6 and branch['source_sft_updates'] == 573
    with pytest.raises(ValueError, match='candidate identity'):
        validate_config({**branch, 'run_id': cfg['run_id']+'-lr5'})


def episode(success, *, task='dev', scenario='recovery', checked=True, submit_attempted=True):
    turns = [
        {'action': {'tool': 'query_sql'}, 'observation': {}, 'new_verified_metrics': ['prior_net_cents']},
        {'action': {'tool': 'calculate_change'}, 'observation': {}, 'new_verified_metrics': ['change_cents']},
    ]
    if checked:
        turns.append({'action': {'tool': 'check_result'}, 'observation': {'passed': True}, 'new_verified_metrics': []})
    if submit_attempted:
        turns.append({'action': {'tool': 'submit'}, 'observation': {'submitted': success}, 'new_verified_metrics': []})
    prefix = {'row': {'new_verified_metrics': ['current_net_cents']}}
    return {'task_id': task, 'scenario': scenario, 'evaluation_mode': 'sample', 'rollout_index': 0,
            'seed': 1, 'start_id': 'prefix', 'sampling': {'temperature': .8}, 'start': {'prefix': [prefix]},
            'turns': turns, 'success': success, 'termination': 'success' if success else 'budget'}


def test_diagnostics_reconstruct_check_and_submit_independently_and_require_pairing():
    w = {'episodes': [episode(True), episode(False, task='other', scenario=None)]}
    b = {'episodes': [episode(False, submit_attempted=False), episode(True, task='other', scenario=None, checked=False)]}
    result = midpoint.completion_diagnostics(w, b)
    recovery = result['standard']['sample_recovery']['lr4']
    assert recovery['all_metrics_certified'] == recovery['check_passed'] == 1
    assert recovery['passed_check_without_submit'] == 1 and recovery['submitted_successfully'] == 0
    complete = result['standard']['sample_complete']['lr4']
    assert complete['check_passed'] == 0 and complete['submitted_successfully'] == 1
    assert result['paired']['sample_recovery']['candidate_losses_with_all_metrics'] == 1
    assert result['paired']['sample_recovery']['candidate_losses_exhausting_budget'] == 1
    b['episodes'][0]['seed'] += 1
    with pytest.raises(ValueError, match='pair differs'):
        midpoint.completion_diagnostics(w, b)


@pytest.mark.parametrize('mode', ['baseline_reject', 'train_reject', 'full_reject', 'full_pass'])
def test_driver_keeps_midpoint_gates_and_exports_only_qualified_b(tmp_path, monkeypatch, mode):
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
        if label == 'gate-lr4':
            atomic_json(tmp_path/'train-gates/lr4.json', {'passed': mode != 'train_reject'})

    def evaluate(child, name, *args):
        if name == 'w':
            return dict(result(first=0), elapsed_seconds=1800, episodes=[episode(True)],
                        evaluation_quality_passed=mode != 'baseline_reject')
        calls.append('evaluate-'+name)
        qualified = mode == 'full_pass'
        return dict(result(full=14 if qualified else 13, target=8, first=9), episodes=[episode(qualified)])

    monkeypatch.setattr(study, 'child', child)
    monkeypatch.setattr(study, 'stage', stage)
    monkeypatch.setattr(study, 'evaluate', evaluate)
    monkeypatch.setattr(worker, 'candidate_source', lambda *a: {'adapter_sha256': 'a'*64, 'source_sft_updates': 573})
    selection = study.execute()
    assert selection['selected_warm_start'] == ('lr4' if mode == 'full_pass' else None)
    assert (tmp_path/'ready-B.json').exists() == (mode == 'full_pass')
    assert ('warm-lr4' in calls) == (mode != 'baseline_reject')
    assert ('evaluate-lr4' in calls) == mode.startswith('full')
    diagnostics = read(tmp_path/'completion-diagnostics.json')
    assert (diagnostics['paired'] is not None) == mode.startswith('full')


def test_diagnostics_audit_rejects_missing_or_changed_results(tmp_path):
    from grpo_dose_worker import save_completion_diagnostics
    w = {'episodes': [episode(True)]}
    with pytest.raises(ValueError, match='missing'):
        save_completion_diagnostics(midpoint, tmp_path, w, None, write=False)
    expected = save_completion_diagnostics(midpoint, tmp_path, w, None, write=True)
    assert save_completion_diagnostics(midpoint, tmp_path, w, None, write=False) == expected
    atomic_json(tmp_path/'completion-diagnostics.json', {})
    with pytest.raises(ValueError, match='differ'):
        save_completion_diagnostics(midpoint, tmp_path, w, None, write=False)
