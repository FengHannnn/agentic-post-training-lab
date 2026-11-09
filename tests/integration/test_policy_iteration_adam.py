import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import score_response, no_update_metrics
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (
    guarded_step, lora_parameters, save_optimizer_checkpoint, restore_optimizer_checkpoint,
    adapter_snapshot, compare_adapter_snapshot, optimizer_snapshot, compare_optimizer_snapshot)
from agentic_post_training.trainers.policy_iteration import import_parent_adam, file_hash


@pytest.fixture
def parent(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(41)
    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, attention_dropout=0.)
    cfg._attn_implementation = 'eager'
    policy = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r=2, lora_alpha=4,
        target_modules=['q_proj', 'v_proj'], lora_dropout=0., init_lora_weights=False, task_type='CAUSAL_LM')).eval()
    reference = copy.deepcopy(policy).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=1e-5, weight_decay=0.)
    # Establish nonzero Adam moments and a parent candidate distinct from SFT.
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = -score_response(policy, [1, 3], [4, 2], grad_enabled=True).mean()
        loss.backward()
        optimizer.step()
    path = tmp_path / 'parent'
    save_optimizer_checkpoint(policy, optimizer, path, 16, 'old-input')
    yield policy, reference, optimizer, path
    torch.set_num_threads(threads)


def test_adam_carry_and_new_old_reference_separation_then_local_resume(parent, tmp_path):
    policy, reference, original_optimizer, parent_path = parent
    before = parameter_digest(policy)
    frozen = parameter_digest(policy, frozen_only=True)
    ref_before = parameter_digest(reference)
    assert before != ref_before
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=1e-5, weight_decay=0.)
    torch.manual_seed(999)
    rng = torch.get_rng_state().clone()
    bridge = import_parent_adam(policy, optimizer, parent_path / 'optimizer.pt',
        expected_hash=file_hash(parent_path / 'optimizer.pt'), old_binding='old-input', new_binding='new-input')
    assert torch.equal(torch.get_rng_state(), rng) and bridge['adam_step_at_start'] == 16
    assert compare_optimizer_snapshot(optimizer, optimizer_snapshot(original_optimizer), 0.)['optimizer_state_passed']
    samples = []
    targets = {}
    cache = {}
    for i in range(8):
        response = [4 + i, 2]
        old = score_response(policy, [1, 3], response, grad_enabled=False)
        new = score_response(policy, [1, 3], response, grad_enabled=True)
        assert no_update_metrics(old, new, 1e-5)['passed']
        sample = RolloutSample('new-group', 'new-group', i, 202610280 + i, [1, 3], response,
                               old.tolist(), old.tolist(), 2, 'eos', 'x', 'x<EOS>')
        samples.append(sample)
        targets['new-group', i] = {'trainer_old_token_log_probs': old.tolist(),
                                   'advantage': 1. if i < 4 else -1., 'reward': float(i < 4)}
        cache['new-group', i] = score_response(reference, [1, 3], response, grad_enabled=False)
    assert any(not torch.equal(cache['new-group', i], torch.tensor(targets['new-group', i]['trainer_old_token_log_probs']))
               for i in range(8))
    config = dict(beta=.02, clip_epsilon=.2, tis_cap=2., max_grad_norm=1., stop_sampled_k3=.05, stop_ratio_clip_fraction=.5)
    assert guarded_step(policy, optimizer, samples, targets, cache, config)['updated']
    checkpoint_policy = copy.deepcopy(policy)
    save_optimizer_checkpoint(policy, optimizer, tmp_path / 'step1', 1, 'new-input')
    expected = guarded_step(policy, optimizer, samples, targets, cache, config)
    expected_adapter, expected_adam = adapter_snapshot(policy), optimizer_snapshot(optimizer)
    resumed_optimizer = torch.optim.AdamW(lora_parameters(checkpoint_policy), lr=1e-5, weight_decay=0.)
    assert restore_optimizer_checkpoint(checkpoint_policy, resumed_optimizer, tmp_path / 'step1', 'new-input') == 1
    resumed = guarded_step(checkpoint_policy, resumed_optimizer, samples, targets, cache, config)
    assert resumed['loss'] == pytest.approx(expected['loss'], abs=1e-7)
    assert compare_adapter_snapshot(checkpoint_policy, expected_adapter, 0.)['passed']
    assert compare_optimizer_snapshot(resumed_optimizer, expected_adam, 0.)['optimizer_state_passed']
    assert all(v['step'].item() == 18 for v in resumed_optimizer.state.values())
    assert parameter_digest(reference) == ref_before and all(p.grad is None for p in reference.parameters())
    assert parameter_digest(policy, frozen_only=True) == frozen


@pytest.mark.parametrize('mutation', ['hash', 'binding', 'order', 'step', 'shape', 'nan', 'hyperparameters'])
def test_bad_parent_adam_rejected_before_loading(parent, tmp_path, mutation):
    policy, _, _, path = parent
    original = path / 'optimizer.pt'
    saved = torch.load(original, map_location='cpu', weights_only=True)
    key = next(iter(saved['optimizer']['state']))
    if mutation == 'binding':
        saved['input_digest'] = 'wrong'
    elif mutation == 'order':
        saved['trainable_names'].reverse()
    elif mutation == 'step':
        saved['optimizer']['state'][key]['step'] += 1
    elif mutation == 'shape':
        saved['optimizer']['state'][key]['exp_avg'] = torch.zeros(1)
    elif mutation == 'nan':
        saved['optimizer']['state'][key]['exp_avg'].fill_(float('nan'))
    elif mutation == 'hyperparameters':
        saved['optimizer']['param_groups'][0]['lr'] = .1
    changed = tmp_path / 'changed.pt'
    torch.save(saved, changed)
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=1e-5, weight_decay=0.)
    with pytest.raises(ValueError):
        import_parent_adam(policy, optimizer, changed, expected_hash='wrong' if mutation == 'hash' else file_hash(changed),
                           old_binding='old-input', new_binding='new-input')
    assert not optimizer.state
