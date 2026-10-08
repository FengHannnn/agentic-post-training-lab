"""Real PEFT on a random tiny Llama; no downloads, no pretrained weights."""

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("transformers")

from peft import PeftModel
from transformers import LlamaConfig, LlamaForCausalLM

from agentic_post_training.data import collate_sft
from agentic_post_training.trainers.sft_lora import attach_lora, evaluate_loss, model_inputs, train_steps


def test_lora_updates_only_adapters_and_survives_reload(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=32, pad_token_id=0, eos_token_id=2,
            attention_dropout=0.0,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(42)
        base = LlamaForCausalLM(config)
        original_base = {name: value.detach().clone() for name, value in base.state_dict().items()}
        model = attach_lora(base, rank=2, alpha=4)
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
        adapters = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        samples = [{
            "input_ids": [1, 5, 7, 2], "attention_mask": [1, 1, 1, 1],
            "labels": [-100, -100, 7, 2], "loss_mask": [0, 0, 1, 1],
        }]
        assert evaluate_loss(model, samples, pad_token_id=0) > 0
        history = train_steps(model, samples, pad_token_id=0, max_steps=2)
        assert len(history) == 2
        assert all(row["response_tokens"] == 2 and row["grad_norm"] > 0 for row in history)
        parameters = dict(model.named_parameters())
        assert all(torch.equal(parameters[name], value) for name, value in frozen.items())
        assert any(not torch.equal(parameters[name], value) for name, value in adapters.items())
        assert all(parameters[name].grad is None for name in frozen)
        model.eval()
        inputs = model_inputs(collate_sft(samples, pad_token_id=0))
        with torch.no_grad():
            expected = model(**inputs).logits
        model.save_pretrained(tmp_path / "adapter")
        fresh = LlamaForCausalLM(config)
        fresh.load_state_dict(original_base)
        restored = PeftModel.from_pretrained(fresh, str(tmp_path / "adapter"))
        restored.eval()
        with torch.no_grad():
            actual = restored(**inputs).logits
        assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
