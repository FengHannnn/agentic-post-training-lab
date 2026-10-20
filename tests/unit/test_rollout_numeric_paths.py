import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("numeric_paths", ROOT / "scripts/diagnose_rollout_numerics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def model():
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    config._attn_implementation = "eager"
    result = LlamaForCausalLM(config).eval().requires_grad_(False)
    result.generate = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate"))
    return result


@pytest.mark.parametrize("prompt", [[1], [1, 7, 8]])
@pytest.mark.parametrize("response", [[2], [9, 10, 2]])
def test_cached_forcing_matches_full_forward_without_generating(model, prompt, response):
    full = module.full_logps(model, prompt, response)
    cached = module.cached_logps(model, prompt, response)
    torch.testing.assert_close(torch.tensor(full), torch.tensor(cached), atol=1e-6, rtol=1e-6)
    assert len(full) == len(response)
    assert all(p.grad is None for p in model.parameters())


def test_cached_path_feeds_only_recorded_previous_tokens(model):
    calls = []
    hook = model.register_forward_pre_hook(lambda model, args, kwargs: calls.append(
        (kwargs["input_ids"].tolist(), kwargs["attention_mask"].shape[1], kwargs["use_cache"])), with_kwargs=True)
    try:
        module.cached_logps(model, [1, 7, 8], [9, 10, 2])
    finally:
        hook.remove()
    assert calls == [([[1, 7, 8]], 3, True), ([[9]], 4, True), ([[10]], 5, True)]


@pytest.mark.parametrize("prompt,response", [([], [2]), ([1], [])])
def test_empty_sequence_is_rejected(model, prompt, response):
    for scorer in (module.full_logps, module.cached_logps):
        with pytest.raises(ValueError): scorer(model, prompt, response)
