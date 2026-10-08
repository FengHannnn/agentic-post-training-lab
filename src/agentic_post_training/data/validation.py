"""Validate SFT targets and exact split isolation before training."""

import json
from collections import Counter

from agentic_post_training.data.sft import SFTExample
from agentic_post_training.environments.calculator import calculate, expression_structure
from agentic_post_training.evaluation.tool_calls import EvalCase, score_output
from agentic_post_training.trainers.sft_lora import check_split


def target_for_example(example: SFTExample) -> EvalCase:
    payload = json.loads(example.response)
    expression = payload["arguments"]["expression"] if payload["type"] == "tool_call" else None
    return EvalCase(
        id=example.id, category="legacy", prompt=example.prompt, system=example.system,
        expected_type=payload["type"], expected_expression=expression,
        expected_result=calculate(expression) if expression is not None else None,
    )


def validate_dataset(
    train: list[SFTExample], development: list[SFTExample],
    targets: list[EvalCase], protected: list[EvalCase],
) -> dict:
    if not train or not development:
        raise ValueError("train and development must be non-empty")
    for samples in (train, development):
        if len({(item.system.strip(), item.prompt.strip()) for item in samples}) != len(samples):
            raise ValueError("duplicate inputs within a split")
    check_split(train, development)
    external = [SFTExample(case.id, case.system, case.prompt, "protected") for case in protected]
    check_split(train + development, external)
    target_map = {case.id: case for case in targets}
    sample_ids = {example.id for example in train + development}
    if len(target_map) != len(targets) or set(target_map) != sample_ids:
        raise ValueError("targets must cover all samples exactly once")
    for example in train + development:
        case = target_map[example.id]
        if (case.system, case.prompt) != (example.system, example.prompt):
            raise ValueError(f"{example.id}: target input differs from SFT input")
        score = score_output(example.response, case)
        if not score["end_to_end_pass"]:
            raise ValueError(f"{example.id}: invalid training target: {score['failure']}")
    expression_sets = []
    for cases in ([target_map[item.id] for item in train], [target_map[item.id] for item in development], protected):
        expression_sets.append({expression_structure(case.expected_expression) for case in cases if case.expected_type == "tool_call"})
    if any(expression_sets[left] & expression_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("arithmetic expression structures overlap across splits")
    result = {}
    for name, samples in (("train", train), ("development", development)):
        selected = [target_map[example.id] for example in samples]
        result[name] = {
            "examples": len(samples),
            "routes": dict(Counter(case.expected_type for case in selected)),
            "categories": dict(Counter(case.category for case in selected)),
        }
    return result
