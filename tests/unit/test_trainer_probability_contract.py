import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import (
    score_response, no_update_metrics, sampler_trainer_difference, pack_trainer_scores)


def tiny_policy():
    torch.manual_seed(42)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    config._attn_implementation = "eager"
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4,
        lora_dropout=0, target_modules=["q_proj", "v_proj"], init_lora_weights=False,
        task_type="CAUSAL_LM"))
    return model.eval()


def test_same_path_ratio_one_and_gradients_only_reach_lora():
    model = tiny_policy()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    old = score_response(model, [1, 7, 8], [9, 10, 2], grad_enabled=False)
    new = score_response(model, [1, 7, 8], [9, 10, 2], grad_enabled=True)
    metrics = no_update_metrics(old, new, 1e-5)
    assert metrics["passed"] and metrics["ratio_requires_grad"]
    assert metrics["max_abs_ratio_minus_one"] < 1e-5
    assert not old.requires_grad and new.requires_grad
    # Backward is used only in this tiny unit test, never in the cloud no-update script.
    new.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_" in n)
    assert all(p.grad is None for n, p in model.named_parameters() if "lora_" not in n)
    assert all(torch.equal(before[name], p) for name, p in model.named_parameters())


def test_sampler_difference_is_not_a_no_update_policy_ratio():
    old = torch.tensor([-0.1, -2.0])
    new = old.clone().requires_grad_()
    q = [-0.12, -2.03]
    assert no_update_metrics(old, new, 1e-5)["passed"]
    diagnostic = sampler_trainer_difference(q, old.tolist())
    assert diagnostic["max_abs_logp_delta"] > .01
    assert not diagnostic["behavior_correction_applied"]


@pytest.mark.parametrize("mutation", ["old_graph", "new_detached", "nan", "dtype", "shape"])
def test_invalid_old_new_contract_is_rejected(mutation):
    old = torch.tensor([-.1, -.2])
    new = old.clone().requires_grad_()
    if mutation == "old_graph": old.requires_grad_()
    elif mutation == "new_detached": new = new.detach()
    elif mutation == "nan": old[0] = float("nan")
    elif mutation == "dtype": new = new.double()
    else: new = new[:1]
    with pytest.raises(ValueError): no_update_metrics(old, new, 1e-5)


def test_actual_new_policy_shift_fails_no_update_gate():
    old = torch.tensor([-.1, -.2])
    new = torch.tensor([-.11, -.2], requires_grad=True)
    assert not no_update_metrics(old, new, 1e-5)["passed"]


def batch_fixture():
    groups, rows = [], []
    for g in range(2):
        group = []
        for i, tokens in enumerate(([5, 0], [5, 6, 0])):
            sample = RolloutSample(str(g), str(g), i, 42+i, [1], list(tokens),
                [-.2]*len(tokens), [-.2]*len(tokens), 0, "eos", "text", "text<EOS>")
            group.append(sample)
            rows.append({"group_id": str(g), "sample_index": i, "response_token_ids": list(tokens),
                "trainer_old_token_log_probs": [-.1]*len(tokens),
                "trainer_new_token_log_probs": [-.1]*len(tokens), "eligible": g == 0})
        groups.append(group)
    return groups, rows


def test_three_score_fields_masks_and_excluded_group_are_separate():
    groups, rows = batch_fixture()
    batch = pack_trainer_scores(groups, list(reversed(rows)), 0)
    mask = batch["response_mask"]
    assert mask[0, 0, 1] and not mask[0, 0, 2]  # EOS==PAD, distinguished by length.
    assert torch.equal(batch["policy_ratios"][mask], torch.ones(int(mask.sum())))
    assert not torch.equal(batch["sampler_behavior_log_probs"][mask], batch["trainer_old_log_probs"][mask])
    assert not batch["eligible_response_mask"][1].any()
    assert batch["eligible_response_mask"][0].sum() == 5


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "tokens", "length", "nonfinite"])
def test_score_batch_rejects_mapping_or_value_corruption(mutation):
    groups, rows = batch_fixture()
    if mutation == "duplicate": rows[1] = copy.deepcopy(rows[0])
    elif mutation == "missing": rows.pop()
    elif mutation == "tokens": rows[0]["response_token_ids"] = [8, 0]
    elif mutation == "length": rows[0]["trainer_old_token_log_probs"].pop()
    else: rows[0]["trainer_new_token_log_probs"][0] = float("nan")
    with pytest.raises(ValueError): pack_trainer_scores(groups, rows, 0)
