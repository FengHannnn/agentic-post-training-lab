"""Factorial arithmetic contrasts with a nested, family-complete small subset."""

import json
import random
from collections import Counter, defaultdict
from itertools import product

from agentic_post_training.data.preferences import PreferenceExample
from agentic_post_training.environments.calculator import calculate
from agentic_post_training.evaluation.tool_calls import SYSTEM_PROMPT

FORMS = ("right_grouped", "right_plain", "left_grouped", "left_plain")
WORDINGS = ("symbolic", "natural", "sequence")


def tool_text(expression):
    return json.dumps({"type": "tool_call", "name": "calculator", "arguments": {"expression": expression}},
                      separators=(",", ":"))


def arithmetic_variants(a, b, c):
    targets = (f"{a}*({b}+{c})", f"{a}*{b}+{c}", f"({b}+{c})*{a}", f"{b}+{c}*{a}")
    wrong = (targets[1], targets[0], targets[3], targets[2])
    prompts = {
        "symbolic": tuple(f"Evaluate {expression}." for expression in targets),
        "natural": (
            f"Multiply {a} by the sum of {b} and {c}.",
            f"Add {c} to the product of {a} and {b}.",
            f"Multiply the sum of {b} and {c} by {a}.",
            f"Add {b} to the product of {c} and {a}.",
        ),
        "sequence": (
            f"First find the sum of {b} and {c}, then multiply {a} by that sum.",
            f"First multiply {a} by {b}, then add {c} to that product.",
            f"First add {b} and {c}, then multiply that sum by {a}.",
            f"First multiply {c} by {a}, then add that product to {b}.",
        ),
    }
    for wording in WORDINGS:
        for form, prompt, expression, rejected in zip(FORMS, prompts[wording], targets, wrong):
            yield form, wording, prompt, expression, rejected


def build_balanced_preferences(baseline_hash):
    triples = {"train": [(12 + i, 47 + 4 * i, 20 + 3 * i) for i in range(12)],
               "development": [(30 + i, 103 + 6 * i, 61 + 5 * i) for i in range(4)]}
    rows = {split: [] for split in triples}
    for split, selected in triples.items():
        for a, b, c in selected:
            family = f"v2-arithmetic-{a}-{b}-{c}"
            for form, wording, prompt, expression, wrong in arithmetic_variants(a, b, c):
                provenance = {"baseline_manifest_sha256": baseline_hash,
                    "chosen_source": "deterministic_template_with_executable_target",
                    "rejected_source": "synthetic_counterexample", "synthetic_rule": "wrong_grouping",
                    "wording": wording, "form": form, "numbers": [a, b, c],
                    "contrast_id": f"{family}-{wording}-{form.split('_')[0]}"}
                rows[split].append(PreferenceExample(f"dpo-v2-{split}-{a}-{b}-{c}-{wording}-{form}",
                    family, SYSTEM_PROMPT, prompt, tool_text(expression), tool_text(wrong), form,
                    "tool_call", expression, calculate(expression), provenance))
    greetings = {"train": ("Howdy", "Greetings", "Hey there", "Hello friend", "Good to see you", "A warm hello"),
                 "development": ("Hi everyone", "Good day")}
    suffixes = ("Hope your week has been pleasant.", "Wishing you a peaceful afternoon.",
                "Glad we have a chance to chat.", "Sending you a friendly wave.")
    for split, roots in greetings.items():
        for index, root in enumerate(roots):
            family = f"v2-greeting-{split}-{index}"
            for variant, suffix in enumerate(suffixes):
                rows[split].append(PreferenceExample(f"dpo-v2-{split}-greeting-{index}-{variant}",
                    family, SYSTEM_PROMPT, f"{root}! {suffix}", '{"type":"final","answer":"Hello!"}',
                    tool_text("1+1"), "greeting", "final", provenance={
                        "baseline_manifest_sha256": baseline_hash,
                        "chosen_source": "deterministic_greeting_template",
                        "rejected_source": "synthetic_counterexample", "synthetic_rule": "wrong_route",
                        "wording": "greeting", "greeting_root": root}))
    rng = random.Random(42)
    arithmetic = sorted({row.family_id for row in rows["train"] if row.expected_type == "tool_call"})
    greetings = sorted({row.family_id for row in rows["train"] if row.expected_type == "final"})
    selected = set(rng.sample(arithmetic, 2) + rng.sample(greetings, 1))
    rows["train_small"] = [row for row in rows["train"] if row.family_id in selected]
    return rows


def audit_balanced_design(rows):
    sizes = {"train": (144, 24), "train_small": (24, 4), "development": (48, 8)}
    summary = {}
    for split, (arithmetic_count, greeting_count) in sizes.items():
        selected = rows[split]
        routes = Counter(row.expected_type for row in selected)
        if routes != {"tool_call": arithmetic_count, "final": greeting_count}:
            raise ValueError(f"{split}: route counts differ from frozen design")
        families = defaultdict(list)
        for row in selected:
            families[row.family_id].append(row)
        for family, members in families.items():
            if members[0].expected_type == "tool_call":
                cells = Counter((row.category, row.provenance["wording"]) for row in members)
                if cells != Counter({cell: 1 for cell in product(FORMS, WORDINGS)}):
                    raise ValueError(f"{family}: incomplete or duplicate factorial cells")
            elif len(members) != 4:
                raise ValueError(f"{family}: incomplete greeting family")
        summary[split] = {"pairs": len(selected), "families": len(families),
            "wordings": dict(Counter(row.provenance["wording"] for row in selected)),
            "grouped": sum(row.category.endswith("_grouped") for row in selected),
            "plain": sum(row.category.endswith("_plain") for row in selected)}
    full = {row.id: row for row in rows["train"]}
    if any(full.get(row.id) != row for row in rows["train_small"]):
        raise ValueError("small subset is not an exact subset of full training data")
    small_families = {row.family_id for row in rows["train_small"]}
    if {row.id for row in rows["train_small"]} != {row.id for row in rows["train"] if row.family_id in small_families}:
        raise ValueError("small subset must contain whole families")
    return summary
