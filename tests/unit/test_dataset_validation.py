import json
from pathlib import Path

import pytest

from agentic_post_training.data import SFTExample, load_examples
from agentic_post_training.data.validation import target_for_example, validate_dataset
from agentic_post_training.evaluation.tool_calls import load_eval_cases


def sample(id, prompt, expression):
    response = json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}})
    return SFTExample(id, "system", prompt, response)


def test_same_expression_in_reworded_development_question_is_rejected():
    train = sample("t", "Multiply 3 and 4", "3*4")
    dev = sample("d", "What is 3 times 4?", "(3 * 4)")
    with pytest.raises(ValueError, match="structures overlap"):
        validate_dataset([train], [dev], [target_for_example(train), target_for_example(dev)], [])


def test_training_label_that_disagrees_with_reference_is_rejected():
    train = sample("t", "Multiply 3 and 4", "3*4")
    dev = sample("d", "Multiply 5 and 6", "5*6")
    targets = [target_for_example(train), target_for_example(dev)]
    wrong = sample("t", train.prompt, "3+4")
    with pytest.raises(ValueError, match="invalid training target"):
        validate_dataset([wrong], [dev], targets, [])


def test_duplicate_training_inputs_are_rejected():
    train = [sample("t1", "Multiply 3 and 4", "3*4"), sample("t2", "Multiply 3 and 4", "3*4")]
    dev = sample("d", "Multiply 5 and 6", "5*6")
    with pytest.raises(ValueError, match="duplicate inputs"):
        validate_dataset(train, [dev], [], [])


def test_v2_bracket_examples_are_valid_and_do_not_leak_original_test():
    root = Path(__file__).resolve().parents[2]
    train = load_examples(root / "data/training/sft_train_expanded_v2.jsonl")
    dev = load_examples(root / "data/training/sft_dev_expanded_v2.jsonl")
    targets = load_eval_cases(root / "data/training/expanded_v2_targets.jsonl")
    protected = load_eval_cases(root / "data/eval/tool_calls_v1.jsonl")
    protected += [target_for_example(item) for item in load_examples(root / "data/examples/sft_eval_en.jsonl")]
    summary = validate_dataset(train, dev, targets, protected)
    assert summary["train"]["routes"] == {"tool_call": 104, "final": 40}
    assert summary["development"]["routes"] == {"tool_call": 24, "final": 8}
    for category in ("multiply_sum_right", "multiply_difference_right", "multiply_sum_left", "multiply_difference_left"):
        assert summary["train"]["categories"][category] == 6
        assert summary["development"]["categories"][category] == 2


def test_expanded_dataset_has_valid_targets_and_no_protected_overlap():
    root = Path(__file__).resolve().parents[2]
    train = load_examples(root / "data/training/sft_train_expanded_v1.jsonl")
    dev = load_examples(root / "data/training/sft_dev_expanded_v1.jsonl")
    targets = load_eval_cases(root / "data/training/expanded_v1_targets.jsonl")
    protected = load_eval_cases(root / "data/eval/tool_calls_v1.jsonl")
    protected += [target_for_example(item) for item in load_examples(root / "data/examples/sft_eval_en.jsonl")]
    summary = validate_dataset(train, dev, targets, protected)
    assert summary["train"]["routes"] == {"tool_call": 80, "final": 40}
    assert summary["development"]["routes"] == {"tool_call": 16, "final": 8}
    assert [json.loads(item.response)["type"] for item in train[:6]] == ["tool_call", "tool_call", "final"] * 2
