import json
from pathlib import Path

import pytest

from agentic_post_training.environments.calculator import calculate
from agentic_post_training.evaluation.tool_calls import EvalCase, load_eval_cases, score_output, summarize


@pytest.fixture
def case():
    return EvalCase("case", "multiply", "Multiply 13 and 27.", "tool_call", "13*27", 351)


def call(expression):
    return json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}})


@pytest.mark.parametrize(("expression", "result"), [("(30+18)/6", 8), ("-8*6", -48), ("1.25+0.75", 2), ("7/2", 3.5)])
def test_calculator_supported_operations(expression, result):
    assert calculate(expression) == result


@pytest.mark.parametrize("expression", ["1/0", "2**8", "abs(-1)", "True+1", "x+1", "[1]", "1e309", "1e12*2", "1" * 129])
def test_calculator_rejects_invalid_or_unbounded_inputs(expression):
    with pytest.raises(ValueError):
        calculate(expression)


def test_correct_call_passes_and_executes(case):
    score = score_output(call(" (13 * 27) "), case)
    assert score["end_to_end_pass"] and score["tool_result"] == 351


def test_matching_result_is_not_enough_for_correct_arguments(case):
    score = score_output(call("351"), case)
    assert score["result_correct"] and not score["arguments_correct"]
    assert not score["end_to_end_pass"]


@pytest.mark.parametrize("text", ['```json\n{}\n```', '{} extra', '{"type":"final","type":"tool_call"}', '{"answer":NaN}'])
def test_invalid_json_is_not_repaired(text, case):
    assert not score_output(text, case)["json_valid"]


def test_wrong_tool_is_schema_failure(case):
    text = call("13*27").replace("calculator", "shell")
    assert score_output(text, case)["failure"] == "invalid_schema"


def test_invalid_expression_never_executes(case):
    score = score_output(call("1/0"), case)
    assert not score["arguments_valid"] and not score["tool_execution_success"]


def test_direct_answer_to_calculation_has_wrong_route(case):
    score = score_output('{"type":"final","answer":"351"}', case)
    assert score["schema_valid"] and not score["route_correct"]


def test_greeting_pass_requires_nonempty_final():
    greeting = EvalCase("g", "greeting", "Hi!", "final")
    assert score_output('{"type":"final","answer":"Hello!"}', greeting)["end_to_end_pass"]
    assert not score_output('{"type":"final","answer":" "}', greeting)["schema_valid"]


def test_summary_uses_tool_case_denominator(case):
    tool = {"expected_type": "tool_call", **score_output(call("13*27"), case)}
    greeting = EvalCase("g", "greeting", "Hi!", "final")
    final = {"expected_type": "final", **score_output('{"type":"final","answer":"Hi"}', greeting)}
    metrics = summarize([tool, final])["metrics"]
    assert metrics["json_valid"]["total"] == 2
    assert metrics["arguments_correct"]["total"] == 1


def test_heldout_reference_answers_are_consistent():
    root = Path(__file__).resolve().parents[2]
    cases = load_eval_cases(root / "data/eval/tool_calls_v1.jsonl")
    assert len(cases) == 20
    assert sum(case.expected_type == "tool_call" for case in cases) == 16
