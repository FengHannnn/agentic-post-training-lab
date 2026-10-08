import json
from pathlib import Path

import pytest

from agentic_post_training.evaluation.brackets import diagnostic_summary
from agentic_post_training.evaluation.tool_calls import load_eval_cases, score_output


def test_diagnostic_pairs_change_meaning_without_changing_numbers():
    root = Path(__file__).resolve().parents[2]
    cases = load_eval_cases(root / "data/eval/bracket_diagnostic_v1.jsonl")
    assert len(cases) == 48
    assert len({case.prompt for case in cases}) == 48
    groups = {}
    for case in cases:
        number_set, wording, form = case.category.split("/")
        groups.setdefault((number_set, wording, form.split("_")[0]), []).append(case)
    assert len(groups) == 24
    for pair in groups.values():
        assert len(pair) == 2 and pair[0].expected_result != pair[1].expected_result


def test_equivalent_wording_output_is_semantic_success_but_strict_failure():
    root = Path(__file__).resolve().parents[2]
    case = load_eval_cases(root / "data/eval/bracket_diagnostic_v1.jsonl")[0]
    text = json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": "(6+5)*2"}})
    score = score_output(text, case)
    assert score["result_correct"] and not score["arguments_correct"]


def test_spacing_control_changes_only_prefix_from_compact_questions():
    root = Path(__file__).resolve().parents[2]
    original = [case for case in load_eval_cases(root / "data/eval/bracket_diagnostic_v1.jsonl") if "/compact/" in case.category]
    control = load_eval_cases(root / "data/eval/bracket_spacing_control_v1.jsonl")
    assert len(original) == len(control) == 12
    for first, second in zip(original, control):
        assert second.prompt == first.prompt.replace("Compute ", "Calculate ", 1)
        assert second.expected_expression == first.expected_expression


def test_one_correct_item_does_not_pass_a_contrast_pair():
    rows = [
        {"expected_type": "tool_call", "wording": "spaced", "number_set": "n", "form": form,
         "pair_id": "n/spaced/right", "semantic_pass": passed, "expected_result": expected,
         "tool_result": result, **{key: passed for key in ("json_valid", "schema_valid", "route_correct", "arguments_valid", "arguments_correct", "tool_execution_success", "result_correct", "end_to_end_pass")}}
        for form, passed, expected, result in (("right_grouped", True, 22, 22), ("right_plain", False, 17, 22))
    ]
    summary = diagnostic_summary(rows)
    assert summary["semantic_pass"]["passed"] == 1
    assert summary["pair_semantic_pass"] == {"passed": 0, "total": 1}
    with pytest.raises(ValueError, match="each diagnostic pair"):
        diagnostic_summary(rows[:1])
