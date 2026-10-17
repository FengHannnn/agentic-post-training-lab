import pytest
import torch

pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rollouts.hf_sampling import sample_response


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1

    @property
    def raw(self):
        return self

    def apply_chat_template(self, messages, **kwargs):
        return [1, 4, 5]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return " ".join(str(token) for token in ids if not skip_special_tokens or token not in (0, 1))


def test_sampling_preserves_rng_weights_and_distinguishes_behavior_from_policy():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            config = LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32, eos_token_id=0, bos_token_id=1, pad_token_id=0,
                attention_dropout=0.0)
            config._attn_implementation = "eager"
            model = LlamaForCausalLM(config).eval().requires_grad_(False)
            before = {name: value.clone() for name, value in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            case = EvalCase("tiny", "greeting", "Hello", "final")
            args = dict(group_id="tiny", sample_index=0, seed=91, max_new_tokens=4, temperature=0.8, top_p=1.0)
            sample, error = sample_response(model, TinyTokenizer(), case, **args)
            repeated, repeated_error = sample_response(model, TinyTokenizer(), case, **args)
            assert sample.response_token_ids == repeated.response_token_ids
            assert error < 1e-4 and repeated_error < 1e-4
            assert not torch.allclose(torch.tensor(sample.policy_token_log_probs), torch.tensor(sample.behavior_token_log_probs), atol=1e-6, rtol=1e-6)
            assert torch.equal(rng, torch.get_rng_state())
            assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
            assert all(value.grad is None for value in model.parameters())
    finally:
        torch.set_num_threads(previous_threads)
