"""DPO on a tiny model: same SFT start, isolated reference, update and reload."""

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from agentic_post_training.trainers.sft_lora import attach_lora, train_steps
from agentic_post_training.trainers.dpo import (
    cache_reference, evaluate_preferences, pair_log_probs, parameter_digest, train_dpo,
)


@pytest.mark.parametrize("chosen_nll_weight", [0.0, 1.0])
def test_dpo_freezes_reference_updates_policy_and_reloads(tmp_path, chosen_nll_weight):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, pad_token_id=0, eos_token_id=2, attention_dropout=0.0)
        config._attn_implementation = "eager"
        base = LlamaForCausalLM(config)
        original = {name: value.detach().clone() for name, value in base.state_dict().items()}
        sft = attach_lora(base, rank=2, alpha=4)
        chosen = {"input_ids": [1, 5, 7, 2], "attention_mask": [1] * 4,
                  "labels": [-100, -100, 7, 2], "loss_mask": [0, 0, 1, 1]}
        rejected = {"input_ids": [1, 5, 8, 9, 2], "attention_mask": [1] * 5,
                    "labels": [-100, -100, 8, 9, 2], "loss_mask": [0, 0, 1, 1, 1]}
        train_steps(sft, [chosen], pad_token_id=0, max_steps=1)
        sft.save_pretrained(tmp_path / "sft", save_embedding_layers=False)

        def fresh():
            result = LlamaForCausalLM(config)
            result.load_state_dict(original)
            return result

        policy = PeftModel.from_pretrained(fresh(), str(tmp_path / "sft"), is_trainable=True)
        reference = PeftModel.from_pretrained(fresh(), str(tmp_path / "sft"), is_trainable=False)
        reference.requires_grad_(False)
        pairs = [[chosen, rejected]]
        policy_initial = parameter_digest(policy)
        reference_initial = parameter_digest(reference)
        assert policy_initial == reference_initial
        frozen = parameter_digest(policy, frozen_only=True)
        cached = cache_reference(reference, pairs, 0)
        assert not cached[0].requires_grad
        before = evaluate_preferences(policy, pairs, cached, 0, 0.1)
        assert abs(before["loss"] - 0.69314718) < 1e-6
        assert abs(before["relative_margin"]) < 1e-6
        # A trainable reference is rejected, rather than silently optimized.
        with pytest.raises(ValueError, match="reference must be frozen"):
            cache_reference(policy, pairs, 0)
        history = train_dpo(policy, pairs, cached, 0, max_steps=4, learning_rate=0.01,
                            chosen_nll_weight=chosen_nll_weight)
        after = evaluate_preferences(policy, pairs, cached, 0, 0.1)
        assert after["relative_margin"] > before["relative_margin"]
        assert after["loss"] < before["loss"]
        assert len(history) == 4 and all(row["grad_norm"] > 0 for row in history)
        assert parameter_digest(reference) == reference_initial
        assert all(not p.requires_grad and p.grad is None for p in reference.parameters())
        assert parameter_digest(policy, frozen_only=True) == frozen
        assert all(p.grad is None for p in policy.parameters() if not p.requires_grad)
        assert parameter_digest(policy) != policy_initial
        policy.eval()
        with torch.no_grad():
            expected = pair_log_probs(policy, pairs[0], 0)
        policy.save_pretrained(tmp_path / "dpo", save_embedding_layers=False)
        restored = PeftModel.from_pretrained(fresh(), str(tmp_path / "dpo"))
        restored.eval()
        with torch.no_grad():
            actual = pair_log_probs(restored, pairs[0], 0)
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
