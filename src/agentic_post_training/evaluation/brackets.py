"""Paired diagnostics distinguish strict AST matches from numeric semantics."""

from collections import defaultdict

from agentic_post_training.evaluation.tool_calls import summarize


def diagnostic_summary(records: list[dict]) -> dict:
    result = summarize(records)
    result["semantic_pass"] = {
        "passed": sum(row["semantic_pass"] for row in records), "total": len(records),
        "rate": sum(row["semantic_pass"] for row in records) / len(records),
    }
    for field in ("wording", "number_set", "form"):
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row)
        result["by_" + field] = {
            key: {
                "total": len(rows), "strict_passed": sum(row["end_to_end_pass"] for row in rows),
                "semantic_passed": sum(row["semantic_pass"] for row in rows),
            }
            for key, rows in sorted(groups.items())
        }
    pairs = defaultdict(list)
    for row in records:
        pairs[row["pair_id"]].append(row)
    for rows in pairs.values():
        if len(rows) != 2 or {row["form"].split("_")[1] for row in rows} != {"grouped", "plain"}:
            raise ValueError("each diagnostic pair must contain grouped and plain forms")
    result["pair_semantic_pass"] = {
        "passed": sum(all(row["semantic_pass"] for row in rows) for rows in pairs.values()),
        "total": len(pairs),
    }
    result["pairs"] = [
        {"pair_id": key, "semantic_pass": all(row["semantic_pass"] for row in rows),
         "expected_results": [row["expected_result"] for row in rows],
         "tool_results": [row["tool_result"] for row in rows]}
        for key, rows in sorted(pairs.items())
    ]
    return result
