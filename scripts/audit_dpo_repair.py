"""Rescore saved repair outputs and independently recompute reports and gates."""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path

import torch

from compare_sft_dpo import file_hash, write_json
from agentic_post_training.data.preferences import semantic_pass
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.scaling import v2_subgroups
from agentic_post_training.evaluation.tool_calls import EvalCase, score_output

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def structural_regressions(before, after):
    """Do not let unchanged success totals hide new malformed outputs."""
    indexed = {row["id"]: row for row in after}
    if len(indexed) != len(after) or len({row["id"] for row in before}) != len(before) or set(indexed) != {row["id"] for row in before}:
        raise ValueError("matching unique IDs required for structural audit")
    return {key: sum(bool(row[key]) and not bool(indexed[row["id"]][key]) for row in before)
            for key in ("json_valid", "schema_valid", "route_correct")}


def audit(output):
    summary = read_json(output / "summary.json")
    run = Path(summary["run"])
    training = read_json(run / "summary.json")
    if (file_hash(run / "summary.json") != summary["training_summary_sha256"]
        or file_hash(run / "adapter/adapter_model.safetensors") != summary["adapter_sha256"]):
        raise ValueError("training checkpoint identity differs")
    for mapping in (summary["source_sha256"], training["source_sha256"]):
        for name, expected in mapping.items():
            candidates = (ROOT / name, output / "source" / Path(name).name)
            if not any(path.exists() and file_hash(path) == expected for path in candidates):
                raise ValueError(f"runtime source snapshot unavailable: {name}")
    records = {}
    for name, expected in summary["prediction_files_sha256"].items():
        if file_hash(output / name) != expected:
            raise ValueError(f"prediction hash differs: {name}")
    for label, suites in summary["models"].items():
        records[label] = {}
        for suite in suites:
            rows = [json.loads(line) for line in (output / f"{label}_{suite}.jsonl").read_text(encoding="utf-8").splitlines()]
            if len(rows) != len({row["id"] for row in rows}):
                raise ValueError("prediction IDs repeat")
            for row in rows:
                case = EvalCase(**{field.name: row[field.name] for field in fields(EvalCase)})
                score = score_output(row["prediction"], case)
                if any(row[key] != value for key, value in score.items()) or row["semantic_pass"] != semantic_pass(score, case.expected_type):
                    raise ValueError(f"saved score differs: {row['id']}")
            records[label][suite] = rows
    comparison_count = 0
    for before in ("sft", "pure_dpo"):
        for suite in summary["comparisons"][before]:
            result, changes = compare_records(records[before][suite], records["anchored_dpo"][suite], paired=suite in ("brackets", "spacing_control"))
            if {key: value for key, value in result.items() if key not in ("sft", "dpo")} != summary["comparisons"][before][suite]:
                raise ValueError("comparison differs")
            for label, key in ((before, "sft"), ("anchored_dpo", "dpo")):
                if suite == "preference_v2_development":
                    result[key]["v2_subgroups"] = v2_subgroups(records[label][suite])
                if result[key] != summary["models"][label][suite]:
                    raise ValueError("model summary differs")
            changes = [{("before_"+key[4:] if key.startswith("sft_") else "after_"+key[4:] if key.startswith("dpo_") else key): value for key, value in row.items()} for row in changes]
            if changes != read_json(output / f"{before}_vs_anchored_{suite}_changes.json"):
                raise ValueError("per-item changes differ")
            comparison_count += 1
    cache_pairs = 0
    if training.get("reference_cache_sha256"):
        path = run / "reference_log_probs.pt"
        if file_hash(path) != training["reference_cache_sha256"]:
            raise ValueError("reference cache hash differs")
        cached = torch.load(path, map_location="cpu", weights_only=True)
        for split, values in cached.items():
            if len(values) != training["audit"][split]["pairs"] or any(value.shape != (2,) or value.requires_grad or not torch.isfinite(value).all().item() for value in values):
                raise ValueError("reference cache values differ")
            for side, key in ((0, "chosen_log_prob"), (1, "rejected_log_prob")):
                if not math.isclose(sum(value[side].item() for value in values)/len(values), training["before"][split][key], abs_tol=1e-6, rel_tol=1e-6):
                    raise ValueError("reference initial metrics differ")
            cache_pairs += len(values)
    model = summary["models"]["anchored_dpo"]
    new = model["preference_v2_development"]
    strict, semantic = new["metrics"]["end_to_end_pass"]["passed"], new["semantic_pass"]["passed"]
    gates = {"new_strict_at_least_sft": strict >= 36, "new_semantic_at_least_sft": semantic >= 44,
        "new_json_schema_route_all_valid": all(new["metrics"][key]["passed"] == 56 for key in ("json_valid", "schema_valid", "route_correct")),
        "legacy_strict_not_lower_than_sft": all(model[suite]["metrics"]["end_to_end_pass"]["passed"] >= bound for suite, bound in (("tool_calls", 19), ("brackets", 33), ("spacing_control", 12), ("preference_v1_development", 7))),
        "new_strict_or_semantic_exceeds_sft": strict > 36 or semantic > 44}
    if gates != summary["acceptance_checks"] or all(gates.values()) != summary["all_acceptance_checks_pass"]:
        raise ValueError("quality gates differ")
    structure = {suite: structural_regressions(records["sft"][suite], records["anchored_dpo"][suite])
                 for suite in records["sft"]}
    no_new_structure_failures = not any(count for group in structure.values() for count in group.values())
    result = {"passed": True, "records_rescored": sum(len(rows) for suites in records.values() for rows in suites.values()),
        "semantic_pass_recomputed": True, "prediction_hashes_verified": len(summary["prediction_files_sha256"]),
        "comparisons_recomputed": comparison_count, "runtime_and_training_source_hashes_verified": True,
        "reference_cache_pairs_verified": cache_pairs, "quality_gates_recomputed": True,
        "all_quality_gates_pass": all(gates.values()),
        "post_audit_structural_regressions": structure,
        "post_audit_no_new_structure_failures": no_new_structure_failures,
        "all_current_admission_checks_pass": all(gates.values()) and no_new_structure_failures}
    write_json(output / "verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重算DPO修复的保存记录与预先验收条件，不加载模型")
    parser.add_argument("--behavior", type=Path, required=True)
    parser.add_argument("--require-admission", action="store_true", help="质量门未通过时退出码为2，阻止晋升候选")
    args = parser.parse_args()
    result = audit(args.behavior.resolve())
    print(json.dumps(result, ensure_ascii=False))
    if args.require_admission and not result["all_current_admission_checks_pass"]:
        raise SystemExit(2)
