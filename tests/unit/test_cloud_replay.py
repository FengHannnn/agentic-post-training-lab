import importlib.util
import math
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location("cloud_replay", Path(__file__).resolve().parents[2] / "scripts/replay_cloud_smoke.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_first_response_and_eos_use_preceding_logits():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = torch.zeros(1, 5, 6)
    logits[0, 2, 4] = 8
    logits[0, 3, 5] = 3
    logits[0, 4, :] = 100  # Last logit predicts beyond the saved sequence; must not be used.
    result = module.response_logps(logits, ids, 3)
    expected = torch.stack((torch.log_softmax(logits[0, 2], -1)[4],
                            torch.log_softmax(logits[0, 3], -1)[5]))
    torch.testing.assert_close(result, expected)
    assert result.shape == (2,)


def test_half_logits_are_scored_in_float32():
    result = module.response_logps(torch.zeros(1, 3, 5, dtype=torch.float16), torch.tensor([[1, 2, 0]]), 2)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.tensor([-math.log(5)]))


def test_gate_detects_wrong_probability_alignment():
    assert module.compare_logps([-0.1, -2.0], [-0.1005, -2.0005], 0.001)["passed"]
    result = module.compare_logps([-0.1, -2.0], [-2.0, -0.1], 0.001)
    assert not result["passed"]
    assert result["sequence_logp_abs_error"] == 0  # Sum alone hides token misalignment.


@pytest.mark.parametrize("saved,replayed,tolerance", [
    ([], [], 0.001), ([-0.1], [-0.1, -0.2], 0.001),
    ([float("nan")], [-0.1], 0.001), ([-0.1], [float("inf")], 0.001),
    ([-0.1], [1.], 0.001), ([-0.1], [-0.1], 0), ([-0.1], [-0.1], float("nan")),
])
def test_reject_invalid_comparison(saved, replayed, tolerance):
    with pytest.raises(ValueError):
        module.compare_logps(saved, replayed, tolerance)


@pytest.mark.parametrize("prompt_length", [0, 3, 4])
def test_reject_missing_prompt_or_response(prompt_length):
    with pytest.raises(ValueError):
        module.response_logps(torch.zeros(1, 3, 5), torch.tensor([[1, 2, 0]]), prompt_length)
