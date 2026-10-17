import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from agentic_post_training.data.demo_tokenizer import DemoChatTokenizer
from agentic_post_training.data.preferences import PreferenceExample


def tool(expression):
    return json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}})


@pytest.fixture
def select(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    function = importlib.import_module("collect_sft_training_errors").select_error
    example = PreferenceExample("train", "family", "Use calculator JSON.", "Multiply two by three.",
        tool("2*3"), tool("2+3"), "arithmetic", "tool_call", "2*3", 6,
        {"wording": "natural", "rejected_source": "synthetic_counterexample"})

    def run(prediction, *, limited=False, max_length=512, row=example):
        generation = {"prediction": prediction, "raw_prediction": prediction + "<eos>",
                      "generated_tokens": 20, "hit_token_limit": limited}
        return function(row, generation, DemoChatTokenizer(), max_length, {"inventory_sha256": "fixed"})
    return example, run


def test_actual_wrong_output_replaces_only_rejected_and_records_origin(select):
    example, run = select
    record, candidate = run(tool("2-3"))
    assert record["selection"] == "candidate"
    assert candidate.rejected == tool("2-3")
    assert candidate.chosen == example.chosen and candidate.family_id == example.family_id
    assert candidate.provenance["original_synthetic_rejected"] == example.rejected
    assert candidate.provenance["rejected_generation_origin"]["inventory_sha256"] == "fixed"
    assert example.provenance["rejected_source"] == "synthetic_counterexample"


@pytest.mark.parametrize("expression,reason", [("2*3", "correct"), ("3*2", "semantic_valid_strict_fail")])
def test_correct_or_numerically_equivalent_outputs_are_not_negatives(select, expression, reason):
    _, run = select
    record, candidate = run(tool(expression))
    assert candidate is None and record["selection"] == reason


@pytest.mark.parametrize("prediction,limited,reason", [
    ("{", True, "hit_token_limit"), (" ", False, "empty_output"),
])
def test_incomplete_or_empty_outputs_are_filtered(select, prediction, limited, reason):
    _, run = select
    record, candidate = run(prediction, limited=limited)
    assert candidate is None and record["selection"] == reason


def test_complete_malformed_json_is_a_valid_error_candidate(select):
    _, run = select
    record, candidate = run("{broken}")
    assert record["failure"] == "invalid_json" and candidate is not None


def test_overlength_error_is_filtered_without_truncation(select):
    _, run = select
    record, candidate = run(tool("2+3"), max_length=20)
    assert record["selection"] == "exceeds_max_length" and candidate is None


def test_invalid_chosen_label_is_not_silently_filtered(select):
    example, run = select
    with pytest.raises(ValueError, match="invalid chosen"):
        run(tool("2-3"), row=replace(example, chosen="{}"))
