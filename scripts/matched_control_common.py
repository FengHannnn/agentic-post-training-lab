"""Pinned data, exposure and admission checks for the three-arm experiment."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from compare_sft_dpo import file_hash
from build_dpo_preferences import protected_cases
from build_matched_preferences import matched_rows
from agentic_post_training.data import load_examples
from agentic_post_training.data.preferences import load_preferences, validate_preferences

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/preferences/dpo_v3_matched_errors"
PLAN_HASH = "a3520bf40a59e07e058b0cb0804571e4d999b197c3ca36de0b110b5601465405"
ARMS = ("chosen_sft", "synthetic_dpo", "real_dpo")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_design(tokenizer=None):
    if file_hash(DATA / "experiment_plan.json") != PLAN_HASH:
        raise ValueError("frozen experiment plan changed")
    plan, manifest = read_json(DATA / "experiment_plan.json"), read_json(DATA / "manifest.json")
    if manifest["plan_sha256"] != PLAN_HASH or manifest["files"] != plan["data_files_sha256"]:
        raise ValueError("plan/data manifest mismatch")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_hash(DATA / name) != expected:
            raise ValueError("frozen dataset changed")
    baseline = ROOT / plan["baseline"]
    if file_hash(baseline / "manifest.json") != plan["baseline_manifest_sha256"]:
        raise ValueError("frozen baseline identity differs")
    real = load_preferences(DATA / "train_real.jsonl")
    synthetic = load_preferences(DATA / "train_synthetic.jsonl")
    chosen = load_examples(DATA / "train_chosen.jsonl")
    rebuilt = matched_rows(real, synthetic)
    if rebuilt["train_chosen"] != chosen or len(real) != plan["common"]["max_steps"]:
        raise ValueError("chosen exposures differ")
    dev = load_preferences(DATA / "development.jsonl")
    if tokenizer is not None:
        protected = protected_cases()
        for name, expected in manifest["protected_v1_data_sha256"].items():
            path = ROOT / "data/preferences/dpo_v1" / name
            if file_hash(path) != expected:
                raise ValueError("protected data changed")
            protected += [row.target() for row in load_preferences(path)]
        import hashlib
        protected_hash = hashlib.sha256(json.dumps([asdict(case) for case in protected],
            ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if protected_hash != manifest["protected_targets_sha256"]:
            raise ValueError("protected targets changed")
        for name, rows in (("train_real", real), ("train_synthetic", synthetic)):
            actual = validate_preferences(rows, dev, protected, tokenizer, plan["common"]["max_length"])
            if actual != manifest["audit"][name]:
                raise ValueError("preference audit differs")
    return plan, manifest, baseline, {"real_dpo": real, "synthetic_dpo": synthetic,
        "chosen_sft": chosen, "development": dev}


def exposure_order(size, seed):
    return torch.randperm(size, generator=torch.Generator().manual_seed(seed)).tolist()


def verify_exposures(histories, examples, plan):
    order = exposure_order(len(examples), plan["common"]["seed"])
    expected = [(step + 1, index, examples[index].id) for step, index in enumerate(order)]
    for label in ARMS:
        rows = histories[label]
        if [(row["step"], row["pair_index"], row["pair_id"]) for row in rows] != expected:
            raise ValueError(f"exposure mismatch: {label}")
        if any(row["learning_rate"] != plan["common"]["learning_rate"] for row in rows):
            raise ValueError("actual learning rate differs")
    for step in range(len(order)):
        if len({histories[label][step]["chosen_response_tokens"] for label in ARMS}) != 1:
            raise ValueError("chosen token exposure differs")
    return True


def pairwise_safety(before, after):
    details = {}
    for suite, rows in before.items():
        indexed = {row["id"]: row for row in after[suite]}
        if len(indexed) != len(after[suite]) or len({row["id"] for row in rows}) != len(rows) or set(indexed) != {row["id"] for row in rows}:
            raise ValueError("matching unique IDs required")
        details[suite] = {key: sum(bool(row[key]) and not bool(indexed[row["id"]][key]) for row in rows)
            for key in ("json_valid", "schema_valid", "route_correct")}
        details[suite]["new_token_limit"] = sum(not row["hit_token_limit"] and indexed[row["id"]]["hit_token_limit"] for row in rows)
    return details


def admission_checks(baseline, candidate, plan, chosen_sft=None):
    def counts(rows):
        return sum(row["end_to_end_pass"] for row in rows), sum(row["semantic_pass"] for row in rows)
    suite = "preference_v2_development"
    strict, semantic = counts(candidate[suite])
    safety = pairwise_safety(baseline, candidate)
    checks = {"new_strict_at_least_baseline": strict >= plan["evaluation"]["baseline_strict"][suite],
        "new_semantic_at_least_baseline": semantic >= plan["evaluation"]["baseline_v2_semantic"],
        "legacy_strict_at_least_baseline": all(counts(candidate[name])[0] >= bound
            for name, bound in plan["evaluation"]["baseline_strict"].items() if name != suite),
        "no_new_structure_or_limits_vs_baseline": not any(value for group in safety.values() for value in group.values())}
    result = {"recovery_checks": checks, "recovery_pass": all(checks.values()), "safety_vs_baseline": safety}
    if chosen_sft is not None:
        other_strict, other_semantic = counts(chosen_sft[suite])
        extra_safety = pairwise_safety(chosen_sft, candidate)
        extra = {"new_strict_at_least_chosen_sft": strict >= other_strict,
            "new_semantic_at_least_chosen_sft": semantic >= other_semantic,
            "new_at_least_one_improvement": strict > other_strict or semantic > other_semantic,
            "legacy_strict_at_least_chosen_sft": all(counts(candidate[name])[0] >= counts(chosen_sft[name])[0] for name in candidate if name != suite),
            "no_new_structure_or_limits_vs_chosen_sft": not any(value for group in extra_safety.values() for value in group.values())}
        result.update(added_value_checks=extra, safety_vs_chosen_sft=extra_safety,
                      conditional_dpo_added_value_pass=all(checks.values()) and all(extra.values()))
    return result
