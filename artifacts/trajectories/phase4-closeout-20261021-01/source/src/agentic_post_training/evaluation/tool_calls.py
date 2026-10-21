import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_post_training.environments.calculator import calculate, expression_structure


SYSTEM_PROMPT = "You are a tool assistant. Output only JSON. For arithmetic, use calculator with an expression argument. For greetings, give a final answer."


@dataclass(frozen=True)
class EvalCase:
    id: str
    category: str
    prompt: str
    expected_type: str
    expected_expression: str | None = None
    expected_result: float | None = None
    system: str = SYSTEM_PROMPT

    def __post_init__(self) -> None:
        for name in ("id", "category", "prompt", "system"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.expected_type not in ("tool_call", "final"):
            raise ValueError("expected_type must be tool_call or final")
        if self.expected_type == "tool_call":
            if type(self.expected_result) not in (int, float):
                raise ValueError("tool cases require a numeric expected_result")
            result = calculate(self.expected_expression)
            if not math.isclose(result, self.expected_result, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("reference expression and result disagree")
        elif self.expected_expression is not None or self.expected_result is not None:
            raise ValueError("final cases must not contain arithmetic targets")


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                case = EvalCase(**json.loads(line))
                if case.id in seen:
                    raise ValueError("duplicate evaluation ID")
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            seen.add(case.id)
            cases.append(case)
    if not cases:
        raise ValueError("evaluation cases must not be empty")
    return cases


def reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def score_output(text: str, case: EvalCase) -> dict[str, Any]:
    score = {
        "json_valid": False, "schema_valid": False, "route_correct": False,
        "arguments_valid": False, "arguments_correct": False,
        "tool_execution_success": False, "result_correct": False,
        "end_to_end_pass": False, "tool_result": None, "failure": None,
    }
    try:
        payload = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (TypeError, ValueError, RecursionError):
        score["failure"] = "invalid_json"
        return score
    score["json_valid"] = True
    if not isinstance(payload, dict):
        score["failure"] = "invalid_schema"
        return score
    kind = payload.get("type")
    if kind == "final":
        valid = set(payload) == {"type", "answer"} and isinstance(payload["answer"], str) and bool(payload["answer"].strip())
    elif kind == "tool_call":
        arguments = payload.get("arguments")
        valid = (
            set(payload) == {"type", "name", "arguments"}
            and payload.get("name") == "calculator"
            and isinstance(arguments, dict) and set(arguments) == {"expression"}
            and isinstance(arguments["expression"], str) and bool(arguments["expression"].strip())
        )
    else:
        valid = False
    if not valid:
        score["failure"] = "invalid_schema"
        return score
    score["schema_valid"] = True
    score["route_correct"] = kind == case.expected_type
    if kind == "final":
        score["end_to_end_pass"] = score["route_correct"]
        score["failure"] = None if score["end_to_end_pass"] else "wrong_route"
        return score
    expression = payload["arguments"]["expression"]
    try:
        result = calculate(expression)
    except ValueError:
        score["failure"] = "invalid_expression"
        return score
    score.update(arguments_valid=True, tool_execution_success=True, tool_result=result)
    if case.expected_type == "tool_call":
        score["arguments_correct"] = expression_structure(expression) == expression_structure(case.expected_expression)
        score["result_correct"] = math.isclose(result, case.expected_result, rel_tol=1e-9, abs_tol=1e-9)
    score["end_to_end_pass"] = score["route_correct"] and score["arguments_correct"] and score["result_correct"]
    score["failure"] = (
        None if score["end_to_end_pass"] else
        "wrong_route" if not score["route_correct"] else
        "wrong_arguments" if not score["arguments_correct"] else "wrong_result"
    )
    return score


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    metrics = {}
    tool_only = {"arguments_valid", "arguments_correct", "tool_execution_success", "result_correct"}
    for metric in ("json_valid", "schema_valid", "route_correct", "arguments_valid", "arguments_correct", "tool_execution_success", "result_correct", "end_to_end_pass"):
        selected = [row for row in records if row["expected_type"] == "tool_call"] if metric in tool_only else records
        passed = sum(bool(row[metric]) for row in selected)
        metrics[metric] = {"passed": passed, "total": len(selected), "rate": passed / len(selected) if selected else None}
    return {"cases": len(records), "metrics": metrics}
