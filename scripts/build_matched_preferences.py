"""Freeze same-input real/synthetic negatives and a bounded three-arm plan."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from build_dpo_preferences import protected_cases
from compare_sft_dpo import file_hash, write_json
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, validate_preferences
from agentic_post_training.data.sft import SFTExample

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baselines/sft-v2-phase3"
COLLECTION = ROOT / "artifacts/evaluations/sft-training-errors-20261017-200708-66935487"
HISTORY = ROOT / "artifacts/evaluations/dpo-v2-scaling-behavior-20261016-163551-cb6e9477"
V2 = ROOT / "data/preferences/dpo_v2"


def matched_rows(candidates, original):
    """Fix inputs/chosen/order across arms; only rejected and its source differ."""
    indexed = {row.id: row for row in original}
    if not candidates or len(indexed) != len(original) or len({row.id for row in candidates}) != len(candidates):
        raise ValueError("non-empty candidates and unique IDs required")
    synthetic, chosen = [], []
    for row in candidates:
        control = indexed.get(row.id)
        if control is None or any(value != asdict(control)[key] for key, value in asdict(row).items()
                                  if key not in ("rejected", "provenance")):
            raise ValueError("matched input, family, chosen or target differs")
        if (row.provenance.get("rejected_source") != "frozen_sft_greedy_training_error"
            or row.provenance.get("original_synthetic_rejected") != control.rejected):
            raise ValueError("candidate rejected provenance differs")
        synthetic.append(control)
        chosen.append(SFTExample(row.id, row.system, row.prompt, row.chosen))
    return {"train_real": candidates, "train_synthetic": synthetic, "train_chosen": chosen}


def jsonl(rows):
    return "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows)


def make_plan(baseline_hash, files, generation, evaluation_hashes, history_hash):
    return {
        "experiment_id": "phase3-matched-errors-38-single-pass",
        "baseline": "artifacts/baselines/sft-v2-phase3", "baseline_manifest_sha256": baseline_hash,
        "preference_data": "data/preferences/dpo_v3_matched_errors", "data_files_sha256": files,
        "common": {"max_steps": 38, "learning_rate": 1e-5, "max_grad_norm": 1.0,
            "seed": 42, "cpu_threads": 2, "max_length": 192, "batch_pairs_or_chosen": 1,
            "optimizer": "AdamW", "weight_decay": 0.0, "scheduler": None},
        "arms": {
            "chosen_sft": {"objective": "response-token-mean chosen CE", "train_file": "train_chosen.jsonl"},
            "synthetic_dpo": {"objective": "pure sigmoid DPO", "train_file": "train_synthetic.jsonl", "beta": 0.1, "chosen_nll_weight": 0.0},
            "real_dpo": {"objective": "pure sigmoid DPO", "train_file": "train_real.jsonl", "beta": 0.1, "chosen_nll_weight": 0.0},
        },
        "order": "torch.randperm(38, generator=torch.Generator().manual_seed(42)); same ID order in every arm",
        "budget": {"training_runs": 3, "steps_per_run": 38, "chosen_exposures_per_arm": 38,
            "same_response_token_compute": False, "report_actual_tokens_and_seconds": True,
            "extra_hyperparameter_trials": 0},
        "evaluation": {"generation": generation,
            "historical_behavior": str(HISTORY.relative_to(ROOT)), "historical_summary_sha256": history_hash,
            "historical_sft_prediction_sha256": evaluation_hashes,
            "suites": {"preference_v2_development": 56, "tool_calls": 20, "brackets": 48,
                "spacing_control": 12, "preference_v1_development": 10},
            "baseline_strict": {"preference_v2_development": 36, "tool_calls": 19,
                "brackets": 33, "spacing_control": 12, "preference_v1_development": 7},
            "baseline_v2_semantic": 44,
            "reuse_sft_requires": "identity/settings/tokenizer/source checks, full rescoring and 3 fresh generation probes",
            "report": ["strict/semantic/JSON/schema/route", "paired arithmetic and greetings separately",
                "per-ID improved/regressed/both_pass/both_fail", "all 3 arm contrasts plus frozen baseline",
                "new per-ID structure failures, even for both_fail examples", "token limits and subgroup failures"],
            "final_unseen_test": False},
        "acceptance": {
            "artifact_invariants": "initial weights equal; only LoRA updates; base/reference unchanged; save/reload logits agree",
            "candidate_recovery": "new strict>=36 and semantic>=44; old suite strict>=baseline; no new per-ID JSON/schema/route failures versus baseline; no new token-limit failures",
            "conditional_dpo_added_value": "DPO must also have new strict and semantic >= chosen_sft with at least one strictly greater; old suite strict >= chosen_sft; no new per-ID structure or token-limit failures versus chosen_sft",
            "source_effect": "compare real_dpo versus synthetic_dpo on identical inputs/chosen/settings; report tradeoffs without claiming universal causality",
            "automatic_baseline_replacement": False},
        "stop_rule": "Run each arm and the common generation comparison once. Record conditional negative results if no DPO added value; do not append LR/beta/anchor sweeps.",
        "limitations": ["单种子、38个错误条件选择输入、仅12个相关family，不是38个独立任务",
            "措辞/形式不平衡且没有问候训练，不声称完整family设计",
            "相同步数和chosen曝光，不是相同token计算量或各算法最优超参数",
            "1e-5为预先选择的保守学习率，不能保证纯DPO或SFT充分学习",
            "只观察一次冻结SFT贪心错误，26个负例文本改变、12个不变",
            "开发集已反复查看且数字范围不同，无最终未查看泛化证据"],
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="冻结38同题真实/合成负例与chosen-only三组设计；不训练")
    parser.add_argument("--output", type=Path, default=ROOT / "data/preferences/dpo_v3_matched_errors")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.audit_only and output.exists():
        raise FileExistsError(f"refusing to overwrite dataset: {output}")
    # Reconstruct candidates from the pinned raw generations, rather than trust a new file.
    subprocess.run([sys.executable, str(ROOT / "scripts/collect_sft_training_errors.py"),
        "--output", str(COLLECTION), "--audit-only"], check=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    collection = json.loads((COLLECTION / "summary.json").read_text(encoding="utf-8"))
    v2_manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    rows = matched_rows(load_preferences(COLLECTION / "candidate_preferences.jsonl"), load_preferences(V2 / "train.jsonl"))
    if len(rows["train_real"]) != 38:
        raise ValueError("frozen design requires exactly 38 candidates")
    rows["development"] = load_preferences(V2 / "development.jsonl")
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(BASELINE / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    protected = protected_cases()
    for name in v2_manifest["protected_v1_data_sha256"]:
        protected += [row.target() for row in load_preferences(ROOT / "data/preferences/dpo_v1" / name)]
    audit = {arm: validate_preferences(rows[arm], rows["development"], protected, tokenizer, 192)
             for arm in ("train_real", "train_synthetic")}
    contents = {f"{name}.jsonl": jsonl(selected) for name, selected in rows.items()}
    files = {name: hashlib.sha256(content.encode("utf-8")).hexdigest() for name, content in contents.items()}
    history = json.loads((HISTORY / "summary.json").read_text(encoding="utf-8"))
    if (history["baseline_manifest_sha256"] != collection["baseline_manifest_sha256"]
        or history["generation"] != baseline["generation"]):
        raise ValueError("historical baseline/generation differs")
    history_hashes = {name: digest for name, digest in history["prediction_files_sha256"].items() if name.startswith("sft_")}
    if len(history_hashes) != 5:
        raise ValueError("five historical SFT suites required")
    for name, digest in history_hashes.items():
        if file_hash(HISTORY / name) != digest:
            raise ValueError("historical baseline predictions changed")
    plan = make_plan(collection["baseline_manifest_sha256"], files, baseline["generation"], history_hashes, file_hash(HISTORY / "summary.json"))
    plan_text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    manifest = {"dataset_id": "dpo-v3-matched-training-errors", "baseline_manifest_sha256": collection["baseline_manifest_sha256"],
        "files": files, "plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(), "max_length": 192,
        "source_collection": str(COLLECTION.relative_to(ROOT)),
        "source_collection_sha256": file_hash(COLLECTION / "summary.json"),
        "source_v2_manifest_sha256": file_hash(V2 / "manifest.json"),
        "builder_sha256": file_hash(Path(__file__)),
        "protected_v1_data_sha256": v2_manifest["protected_v1_data_sha256"],
        "protected_targets_sha256": v2_manifest["protected_targets_sha256"],
        "design": {"pairs_per_dpo_arm": 38, "chosen_examples": 38, "development": 56,
            "families": len({row.family_id for row in rows["train_real"]}),
            "wordings": dict(Counter(row.provenance["wording"] for row in rows["train_real"])),
            "same_rejected_text": sum(a.rejected == b.rejected for a, b in zip(rows["train_real"], rows["train_synthetic"]))},
        "audit": audit, "data_audit_passed": True, "training_started": False,
        "runner_adaptation_required": True}
    if args.audit_only:
        if manifest != json.loads((output / "manifest.json").read_text(encoding="utf-8")):
            raise ValueError("saved manifest differs from reconstruction")
        for name, content in {**contents, "experiment_plan.json": plan_text}.items():
            if (output / name).read_bytes() != content.encode("utf-8"):
                raise ValueError(f"saved data/plan differs: {name}")
    else:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in {**contents, "experiment_plan.json": plan_text}.items():
            (output / name).write_text(content, encoding="utf-8", newline="\n")
        write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "design": manifest["design"], "audit_passed": True,
        "plan_sha256": manifest["plan_sha256"], "training_started": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
