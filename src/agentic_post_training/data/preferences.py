"""Small preference pairs with executable targets and split isolation checks."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agentic_post_training.data.sft import SFTExample, encode_sft
from agentic_post_training.environments.calculator import expression_structure
from agentic_post_training.evaluation.tool_calls import EvalCase, score_output


@dataclass(frozen=True)
class PreferenceExample:
    id: str
    family_id: str
    system: str
    prompt: str
    chosen: str
    rejected: str
    category: str
    expected_type: str
    expected_expression: str | None = None
    expected_result: float | None = None
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("id", "family_id", "system", "prompt", "chosen", "rejected", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.chosen.strip() == self.rejected.strip():
            raise ValueError("chosen and rejected must differ")
        if not isinstance(self.provenance, dict):
            raise ValueError("provenance must be an object")
        self.target()

    def target(self):
        return EvalCase(self.id, self.category, self.prompt, self.expected_type,
                        self.expected_expression, self.expected_result, self.system)


def load_preferences(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(PreferenceExample(**json.loads(line)))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{index}: {error}") from error
    if not rows:
        raise ValueError("preference dataset must be non-empty")
    if len({row.id for row in rows}) != len(rows):
        raise ValueError("duplicate preference ID")
    return rows


def semantic_pass(score, kind):
    return score["schema_valid"] and score["route_correct"] and (
        score["result_correct"] if kind == "tool_call" else score["end_to_end_pass"])


def validate_preferences(train, development, protected, tokenizer=None, max_length=192):
    if not train or not development:
        raise ValueError("train and development must be non-empty")
    if {row.family_id for row in train} & {row.family_id for row in development}:
        raise ValueError("family overlap across splits")
    groups = (train, development, protected)
    keys = []
    for rows in groups:
        keys.append((
            {row.id for row in rows},
            {(row.system.strip(), row.prompt.strip()) for row in rows},
            {row.prompt.strip() for row in rows},
            {expression_structure(row.expected_expression) for row in rows
             if row.expected_type == "tool_call"},
        ))
    for left, right in ((0, 1), (0, 2), (1, 2)):
        for name, a, b in zip(("ID", "input", "prompt", "expression"), keys[left], keys[right]):
            if a & b:
                raise ValueError(f"{name} overlap across preference/protected splits")
    summary = {}
    for split, rows in (("train", train), ("development", development)):
        if len({row.id for row in rows}) != len(rows) or len({row.prompt.strip() for row in rows}) != len(rows):
            raise ValueError("duplicate ID or prompt within split")
        lengths = {"chosen": [], "rejected": []}
        responses = {"chosen": [], "rejected": []}
        for row in rows:
            case = row.target()
            if not score_output(row.chosen, case)["end_to_end_pass"]:
                raise ValueError(f"{row.id}: chosen is not a correct strict target")
            if semantic_pass(score_output(row.rejected, case), case.expected_type):
                raise ValueError(f"{row.id}: rejected is semantically valid; preference is unjustified")
            for side in lengths:
                if tokenizer is not None:
                    encoded = encode_sft(SFTExample(row.id, row.system, row.prompt, getattr(row, side)),
                                         tokenizer, max_length=max_length)
                    lengths[side].append(len(encoded["input_ids"]))
                    responses[side].append(sum(encoded["loss_mask"]))
        summary[split] = {
            "pairs": len(rows), "families": len({row.family_id for row in rows}),
            "routes": dict(Counter(row.expected_type for row in rows)),
            "categories": dict(Counter(row.category for row in rows)),
            "rejected_sources": dict(Counter(row.provenance.get("rejected_source", "unspecified") for row in rows)),
            "rejected_failures": dict(Counter(score_output(row.rejected, row.target())["failure"] for row in rows)),
        }
        if tokenizer is not None:
            summary[split]["tokens"] = {side: {
                "max_total": max(lengths[side]), "min_response": min(responses[side]),
                "mean_response": sum(responses[side]) / len(rows),
            } for side in lengths}
    return summary
