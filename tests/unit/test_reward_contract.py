import copy
import json

import pytest

from agentic_post_training.evaluation.tool_calls import EvalCase, score_output
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rewards.audit import audit_records


CASE = EvalCase("math", "arithmetic", "Multiply 3 by the sum of 4 and 5.", "tool_call", "3*(4+5)", 27)


def tool(expression):
    return json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}})


@pytest.mark.parametrize("text,expected,semantic", [
    (tool("3*(4+5)"), 1, True),
    (tool(" (3) * ((4 + 5)) "), 1, True),
    (tool("(4+5)*3"), 0, True),
    (tool("27"), 0, True),
    (tool("3*4+5"), 0, False),
    (tool("1/0"), 0, False),
    (tool("__import__('os')"), 0, False),
    ("", 0, False),
    ("```json\n" + tool("3*(4+5)") + "\n```", 0, False),
    ('{"type":"final","answer":"27"}', 0, False),
    ('{"type":"final","type":"tool_call"}', 0, False),
    ('{"type":"final","answer":NaN}', 0, False),
    (tool("3*(4+5)").replace('"calculator"', '"shell"'), 0, False),
    (tool("3*(4+5)")[:-1] + ',"extra":1}', 0, False),
    ('{"type":"tool_call"', 0, False),
])
def test_reward_counterexamples(text, expected, semantic):
    result = score_reward(text, CASE, finish_reason="eos")
    assert result["reward"] == expected
    assert result["semantic_pass"] is semantic


def test_truncation_and_greeting_scope():
    result = score_reward(tool("3*(4+5)"), CASE, finish_reason="length")
    assert result["reward"] == 0 and result["reason"] == "truncated"
    case = EvalCase("hello", "greeting", "Hello", "final")
    result = score_reward('{"type":"final","answer":"unrelated"}', case, finish_reason="eos")
    assert result["strict_pass"] and not result["eligible"] and result["reward"] == 0
    with pytest.raises(ValueError):
        score_reward("", CASE, finish_reason="unknown")


def records():
    from dataclasses import asdict
    result = []
    for i, expression in enumerate(("3*(4+5)", "3*4+5")):
        text = tool(expression)
        score = score_output(text, CASE)
        result.append({"case": asdict(CASE), "score": score, "semantic_pass": score["result_correct"],
            "sample": {"group_id": "math", "case_id": "math", "sample_index": i, "seed": i,
                "prompt_token_ids": [1], "response_token_ids": [5, 0], "eos_token_id": 0,
                "policy_token_log_probs": [-1., -1.], "behavior_token_log_probs": [-1., -1.],
                "finish_reason": "eos", "text": text, "raw_text": text}})
    return result


def test_signal_and_order():
    report = audit_records(list(reversed(records())))
    assert report["groups"][0]["advantages"] == [1., -1.]
    assert report["has_reward_learning_signal"]
    rows = records()
    rows[1] = copy.deepcopy(rows[0])
    rows[1]["sample"]["sample_index"] = 1
    report = audit_records(rows)
    assert report["groups"][0]["advantages"] == [0., 0.]
    assert report["zero_variance_rate"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "target", "score", "identity", "prompt", "eos"])
def test_reject_corrupt_records(mutation):
    rows = records()
    if mutation == "duplicate":
        rows[1]["sample"]["sample_index"] = 0
    elif mutation == "target":
        rows[1]["case"]["prompt"] = "Another prompt"
    elif mutation == "score":
        rows[1]["score"]["end_to_end_pass"] = True
    elif mutation == "identity":
        rows[1]["sample"]["case_id"] = "other"
    elif mutation == "prompt":
        rows[1]["sample"]["prompt_token_ids"] = [2]
    else:
        rows[1]["sample"]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        audit_records(rows)
