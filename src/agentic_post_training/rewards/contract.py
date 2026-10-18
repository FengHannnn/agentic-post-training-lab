"""Versioned, conservative arithmetic reward; no model or text repair."""

from agentic_post_training.evaluation.tool_calls import EvalCase, score_output

CONTRACT_VERSION = "arithmetic-strict-v1"


def score_reward(text: str, case: EvalCase, *, finish_reason: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    if finish_reason not in ("eos", "length"):
        raise ValueError("unsupported finish reason")
    score = score_output(text, case)
    eligible = case.expected_type == "tool_call"
    semantic = bool(score["schema_valid"] and score["route_correct"] and (
        score["result_correct"] if eligible else score["end_to_end_pass"]))
    passed = eligible and finish_reason == "eos" and score["end_to_end_pass"]
    reason = ("out_of_scope" if not eligible else "truncated" if finish_reason != "eos"
              else "passed" if passed else score["failure"])
    return {"contract_version": CONTRACT_VERSION, "eligible": eligible,
            "reward": float(passed), "reason": reason,
            "strict_pass": score["end_to_end_pass"], "semantic_pass": semantic,
            "semantic_only": semantic and not score["end_to_end_pass"], "score": score}
