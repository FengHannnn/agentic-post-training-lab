import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cloud_smoke", Path(__file__).resolve().parents[2] / "scripts/cloud_model_smoke.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_real_eos_and_length_endings():
    module.validate_response([1], [5, 0], [-0.1, -0.2], 0, "eos")
    module.validate_response([1], [5], [-0.1], 0, "length")


@pytest.mark.parametrize("response,logps,finish", [
    ([], [], "length"), ([5, 0], [-0.1], "eos"),
    ([5, 0], [-0.1, float("nan")], "eos"),
    ([5, 0], [-0.1, 1.0], "eos"),
    ([0, 5], [-0.1, -0.1], "length"),
    ([5, 0], [-0.1, -0.1], "length"),
    ([5], [-0.1], "stop"),
])
def test_reject_malformed_engine_output(response, logps, finish):
    with pytest.raises(ValueError):
        module.validate_response([1], response, logps, 0, finish)
