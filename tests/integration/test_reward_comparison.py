import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from reward_comparison_common import admit_rewards, import_comparison_adam
import run_reward_comparison as driver
from tests.integration.test_policy_iteration_adam import parent
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import score_response
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (guarded_step, lora_parameters, save_optimizer_checkpoint,
    restore_optimizer_checkpoint, optimizer_snapshot, compare_optimizer_snapshot)
from agentic_post_training.trainers.policy_iteration import file_hash


@pytest.fixture
def parent31(parent, tmp_path):
    policy, reference, optimizer, path = parent
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        (-score_response(policy, [1, 3], [4, 2], grad_enabled=True).mean()).backward()
        optimizer.step()
    path = tmp_path / 'parent31'
    save_optimizer_checkpoint(policy, optimizer, path, 15, 'parent-binding')
    return policy, reference, optimizer, path


def test_two_reward_arms_share_parent_adam_and_inputs_then_resume_graded_exactly(parent31, tmp_path):
    original, reference, inherited, path = parent31
    initial, frozen = parameter_digest(original), parameter_digest(original, frozen_only=True)
    arms = {name: copy.deepcopy(original) for name in ('equivalent_binary', 'graded')}
    optimizers = {}
    samples, reference_cache, old_scores = [], {}, {}
    for i in range(32):
        response = [4 + i % 20, 2]
        old = score_response(original, [1, 3], response, grad_enabled=False).tolist()
        samples.append(RolloutSample('g', 'g', i, 500+i, [1, 3], response, old, old, 2, 'eos', 'x', 'x<eos>'))
        old_scores['g', i] = old
        reference_cache['g', i] = score_response(reference, [1, 3], response, grad_enabled=False)
    for arm, policy in arms.items():
        optimizer = torch.optim.AdamW(lora_parameters(policy), lr=1e-5, weight_decay=0.)
        rng = torch.get_rng_state().clone()
        bridge = import_comparison_adam(policy, optimizer, path/'optimizer.pt', expected_hash=file_hash(path/'optimizer.pt'),
            old_binding='parent-binding', new_binding=arm)
        assert torch.equal(rng, torch.get_rng_state()) and not bridge['parent_rng_restored']
        assert parameter_digest(policy) == initial
        assert compare_optimizer_snapshot(optimizer, optimizer_snapshot(inherited), 0.)['optimizer_state_passed']
        optimizers[arm] = optimizer
    targets = {}
    config = dict(beta=.02, clip_epsilon=.2, tis_cap=2., max_grad_norm=1., stop_sampled_k3=.05, stop_ratio_clip_fraction=.5)
    for arm in arms:
        values = [float(i % 4 == 3) if arm == 'equivalent_binary' else (0., .1, .3, 1.)[i % 4] for i in range(32)]
        advantages = admit_rewards(values, arm)
        targets[arm] = {('g', i): {'trainer_old_token_log_probs': old_scores['g', i], 'advantage': advantages[i], 'reward': values[i]} for i in range(32)}
        assert guarded_step(arms[arm], optimizers[arm], samples, targets[arm], reference_cache, config)['updated']
        assert all(v['step'].item() == 32 for v in optimizers[arm].state.values())
        assert parameter_digest(arms[arm], frozen_only=True) == frozen
    assert parameter_digest(arms['equivalent_binary']) != parameter_digest(arms['graded'])
    checkpoint = tmp_path / 'step1'
    policy, optimizer = arms['graded'], optimizers['graded']
    save_optimizer_checkpoint(policy, optimizer, checkpoint, 1, 'graded')
    reloaded = copy.deepcopy(policy)
    expected = guarded_step(policy, optimizer, samples, targets['graded'], reference_cache, config)
    restored = torch.optim.AdamW(lora_parameters(reloaded), lr=1e-5, weight_decay=0.)
    assert restore_optimizer_checkpoint(reloaded, restored, checkpoint, 'graded') == 1
    actual = guarded_step(reloaded, restored, samples, targets['graded'], reference_cache, config)
    assert actual['loss'] == pytest.approx(expected['loss'], abs=1e-8)
    assert parameter_digest(reloaded) == parameter_digest(policy)
    assert compare_optimizer_snapshot(restored, optimizer_snapshot(optimizer), 0.)['optimizer_state_passed']
    assert all(v['step'].item() == 33 for v in restored.state.values())


@pytest.mark.parametrize('mutation', ['local_step', 'adam_step', 'shape', 'nan', 'order', 'hyperparameter', 'binding', 'hash'])
def test_parent_local_and_adam_counters_and_moments_validated_before_import(parent31, tmp_path, mutation):
    policy, _, _, folder = parent31
    saved = torch.load(folder/'optimizer.pt', map_location='cpu', weights_only=True)
    state = saved['optimizer']['state'][next(iter(saved['optimizer']['state']))]
    if mutation == 'local_step': saved['step'] = 31
    elif mutation == 'adam_step': state['step'].fill_(15)
    elif mutation == 'shape': state['exp_avg'] = torch.zeros(1)
    elif mutation == 'nan': state['exp_avg_sq'].fill_(float('nan'))
    elif mutation == 'order': saved['trainable_names'].reverse()
    elif mutation == 'hyperparameter': saved['optimizer']['param_groups'][0]['lr'] = .1
    elif mutation == 'binding': saved['input_digest'] = 'wrong'
    changed = tmp_path/'changed.pt'
    torch.save(saved, changed)
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=1e-5, weight_decay=0.)
    with pytest.raises(ValueError):
        import_comparison_adam(policy, optimizer, changed, expected_hash='bad' if mutation == 'hash' else file_hash(changed),
            old_binding='parent-binding', new_binding='new')
    assert not optimizer.state


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -.1, .2, True])
def test_only_frozen_finite_graded_levels_admitted(bad):
    values = [.1] * 32
    assert admit_rewards(values, 'graded') == [0.] * 32
    values[0] = bad
    with pytest.raises(ValueError): admit_rewards(values, 'graded')
    with pytest.raises(ValueError): admit_rewards([.1] * 32, 'equivalent_binary')
    with pytest.raises(ValueError): admit_rewards([0.] * 8, 'graded')


def test_driver_stops_after_first_failed_arm_and_packs_actual_evidence(tmp_path, monkeypatch):
    output, archive = tmp_path/'run', tmp_path/'review.zip'
    monkeypatch.setattr(driver, 'OUTPUT', output)
    monkeypatch.setattr(driver, 'ARCHIVE', archive)
    monkeypatch.setattr(driver, 'read', lambda p: {'files': {}, 'required_existing_files': {}})
    monkeypatch.setattr(driver, 'verify_files', lambda *a: None)
    monkeypatch.setattr(driver, 'preflight', lambda: {'passed': True})
    calls = []
    class Process:
        def __init__(self, command, **kwargs): calls.append(command[3])
        def wait(self, timeout): return 0 if calls[-1] == 'prepare' else 1
    monkeypatch.setattr(driver.subprocess, 'Popen', Process)
    with pytest.raises(RuntimeError, match='equivalent_binary failed'):
        driver.dispatch(2.18)
    assert calls == ['prepare', 'equivalent_binary']
    assert archive.exists() and (output/'cost_and_run.json').exists()
