"""Separate chosen probability changes from relative preference improvements."""

from collections import defaultdict
import math


def summarize_preference_changes(rows):
    if not rows:
        raise ValueError("diagnostic rows must be non-empty")
    fields = ("delta_chosen", "delta_rejected", "relative_margin", "loss",
              "sft_chosen_log_prob", "dpo_chosen_log_prob",
              "sft_rejected_log_prob", "dpo_rejected_log_prob")
    if any(not math.isfinite(row[key]) for row in rows for key in fields):
        raise ValueError("diagnostic scores must be finite")
    result = {"pairs": len(rows), **{key: sum(row[key] for row in rows) / len(rows) for key in fields},
              "relative_improved": sum(row["relative_margin"] > 0 for row in rows),
              "chosen_probability_increased": sum(row["delta_chosen"] > 0 for row in rows),
              "chosen_decreased_but_relative_improved": sum(
                  row["delta_chosen"] < 0 and row["relative_margin"] > 0 for row in rows)}
    return result


def grouped_preference_changes(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {key: summarize_preference_changes(selected) for key, selected in sorted(groups.items())}
