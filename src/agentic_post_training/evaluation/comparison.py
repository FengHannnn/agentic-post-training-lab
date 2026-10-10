"""Align behavior predictions by ID and retain metric-specific regressions."""

from collections import Counter

from agentic_post_training.evaluation.brackets import diagnostic_summary
from agentic_post_training.evaluation.tool_calls import summarize


def transition(before, after):
    if before and after:
        return "both_pass"
    if not before and not after:
        return "both_fail"
    return "improved" if after else "regressed"


def compare_records(sft, dpo, paired=False):
    maps = [{row["id"]: row for row in rows} for rows in (sft, dpo)]
    if not sft or any(len(mapping) != len(rows) for mapping, rows in zip(maps, (sft, dpo))):
        raise ValueError("non-empty records and unique IDs required")
    if set(maps[0]) != set(maps[1]):
        raise ValueError("comparison ID sets differ")
    changes = []
    for left in sft:
        right = maps[1][left["id"]]
        for key in ("system", "prompt", "expected_type", "expected_expression", "expected_result"):
            if left.get(key) != right.get(key):
                raise ValueError(f"{left['id']}: comparison inputs/targets differ")
        changes.append({
            "id": left["id"], "category": left["category"], "prompt": left["prompt"],
            "expected_expression": left["expected_expression"], "expected_result": left["expected_result"],
            "sft_prediction": left["prediction"], "dpo_prediction": right["prediction"],
            "sft_failure": left["failure"], "dpo_failure": right["failure"],
            "sft_tool_result": left["tool_result"], "dpo_tool_result": right["tool_result"],
            "prediction_changed": left["prediction"] != right["prediction"],
            **{metric: transition(left[metric], right[metric])
               for metric in ("end_to_end_pass", "semantic_pass", "json_valid", "route_correct")},
        })
    summaries = {}
    for label, rows in (("sft", sft), ("dpo", dpo)):
        summaries[label] = diagnostic_summary(rows) if paired else summarize(rows)
        if not paired:
            summaries[label]["semantic_pass"] = {
                "passed": sum(row["semantic_pass"] for row in rows), "total": len(rows)}
        summaries[label]["hit_token_limit_count"] = sum(row["hit_token_limit"] for row in rows)
        summaries[label]["by_category"] = {category: summarize([row for row in rows if row["category"] == category])
                                            for category in sorted({row["category"] for row in rows})}
    transitions = {metric: {key: counts.get(key, 0) for key in ("improved", "regressed", "both_pass", "both_fail")}
                   for metric in ("end_to_end_pass", "semantic_pass", "json_valid", "route_correct")
                   for counts in [Counter(row[metric] for row in changes)]}
    return {**summaries, "transitions": transitions,
            "prediction_changed_count": sum(row["prediction_changed"] for row in changes)}, changes
