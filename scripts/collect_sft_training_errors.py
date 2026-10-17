"""Mine frozen-SFT errors on v2 training inputs; no training or dev generation."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from build_dpo_preferences import protected_cases
from compare_sft_dpo import file_hash, write_json
from freeze_sft_baseline import verify
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import (
    load_preferences, load_preference_splits, semantic_pass, validate_preferences,
)
from agentic_post_training.data.sft import SFTExample, encode_sft
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.scaling import GENERATION_FIELDS
from agentic_post_training.evaluation.tool_calls import score_output

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baselines/sft-v2-phase3"
DATA = ROOT / "data/preferences/dpo_v2"


def select_error(example, generation, tokenizer, max_length, origin):
    """Keep only complete, encodable, semantically wrong generated answers."""
    score = score_output(generation["prediction"], example.target())
    record = {**asdict(example.target()), **{key: generation[key] for key in GENERATION_FIELDS},
              **score, "semantic_pass": semantic_pass(score, example.expected_type),
              "family_id": example.family_id, "wording": example.provenance["wording"]}
    candidate = None
    if generation["hit_token_limit"]:
        reason = "hit_token_limit"
    elif not generation["prediction"].strip():
        reason = "empty_output"
    elif record["semantic_pass"]:
        reason = "correct" if score["end_to_end_pass"] else "semantic_valid_strict_fail"
    else:
        candidate = replace(example, rejected=generation["prediction"], provenance={
            **example.provenance, "rejected_source": "frozen_sft_greedy_training_error",
            "synthetic_rule": None, "original_synthetic_rejected": example.rejected,
            "rejected_generation_origin": origin, "rejected_failure": score["failure"],
        })
        # Never filter a broken chosen label silently.
        if not score_output(candidate.chosen, candidate.target())["end_to_end_pass"]:
            raise ValueError(f"invalid chosen: {example.id}")
        encoded = {}
        try:
            for side in ("chosen", "rejected"):
                encoded[side] = encode_sft(SFTExample(example.id, example.system, example.prompt,
                    getattr(candidate, side)), tokenizer, max_length=max_length)
        except ValueError as error:
            if "exceeds max_length=" not in str(error):
                raise
            reason, candidate = "exceeds_max_length", None
        else:
            reason = "candidate"
            record["tokens"] = {side: {"total": len(value["input_ids"]),
                "response": sum(value["loss_mask"])} for side, value in encoded.items()}
    record["selection"] = reason
    return record, candidate


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="仅在v2训练输入上采集冻结SFT真实错误；不训练")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if (args.resume or args.audit_only) and args.output is None:
        parser.error("resume/audit-only requires --output")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import torch
    from peft import PeftModel

    verify(BASELINE)
    baseline_manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    data_manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    baseline_hash = file_hash(BASELINE / "manifest.json")
    if data_manifest["baseline_manifest_sha256"] != baseline_hash:
        raise ValueError("baseline/data mismatch")
    samples = load_preference_splits(DATA, data_manifest)
    protected = protected_cases()
    for name, digest in data_manifest["protected_v1_data_sha256"].items():
        path = ROOT / "data/preferences/dpo_v1" / name
        if file_hash(path) != digest:
            raise ValueError("protected v1 data changed")
        protected += [row.target() for row in load_preferences(path)]
    import hashlib
    protected_hash = hashlib.sha256(json.dumps([asdict(case) for case in protected],
        ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if protected_hash != data_manifest["protected_targets_sha256"]:
        raise ValueError("protected targets changed")
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(BASELINE / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    max_length = data_manifest["max_length"]
    validate_preferences(samples["train"], samples["development"], protected, tokenizer, max_length)
    settings = baseline_manifest["generation"]
    sources = [Path(__file__), ROOT / "src/agentic_post_training/evaluation/generation.py",
        ROOT / "src/agentic_post_training/evaluation/tool_calls.py",
        ROOT / "src/agentic_post_training/data/preferences.py", ROOT / "src/agentic_post_training/data/sft.py"]
    inventory = {"baseline_manifest_sha256": baseline_hash, "data_manifest_sha256": file_hash(DATA / "manifest.json"),
        "model": baseline_manifest["model"], "resolved_revision": baseline_manifest["resolved_revision"],
        "adapter_sha256": file_hash(BASELINE / "checkpoint/adapter/adapter_model.safetensors"),
        "generation": settings, "max_length": max_length, "input_split": "train.jsonl",
        "development_used_for_generation": False,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources}}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else ROOT / "artifacts/evaluations" / f"sft-training-errors-{stamp}-{uuid4().hex[:8]}"
    if args.resume or args.audit_only:
        if json.loads((output / "run_manifest.json").read_text(encoding="utf-8")) != inventory:
            raise ValueError("collection inventory changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", inventory)
    origin = {"inventory_sha256": file_hash(output / "run_manifest.json"), "prediction_file": "sft_train.jsonl"}
    path = output / "sft_train.jsonl"
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    if len(saved) > len(samples["train"]) or [row["id"] for row in saved] != [row.id for row in samples["train"][:len(saved)]]:
        raise ValueError("saved records must be an ordered unique prefix of training inputs")
    if args.audit_only and len(saved) != len(samples["train"]):
        raise ValueError("collection is incomplete")
    model = None
    if len(saved) < len(samples["train"]):
        set_seed(settings["seed"])
        torch.set_num_threads(settings["cpu_threads"])
        base = AutoModelForCausalLM.from_pretrained(inventory["model"], revision=inventory["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()
        model = PeftModel.from_pretrained(base, str(BASELINE / "checkpoint/adapter"), is_trainable=False)
        model.requires_grad_(False)
        model.eval()
    print(json.dumps({"output": str(output), "training_inputs": len(samples["train"])}, ensure_ascii=False), flush=True)
    records, candidates = [], []
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for index, example in enumerate(samples["train"]):
            generation = saved[index] if index < len(saved) else generate_prediction(model, tokenizer, example.target(), settings["max_new_tokens"])
            record, candidate = select_error(example, generation, tokenizer, max_length, origin)
            if index < len(saved):
                if record != saved[index]:
                    raise ValueError(f"saved score/selection differs: {example.id}")
            else:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{index+1}/{len(samples['train'])}: {record['selection']} ({record['failure']})", flush=True)
            records.append(record)
            if candidate is not None:
                candidates.append(candidate)
    audit = validate_preferences(candidates, samples["development"], protected, tokenizer, max_length) if candidates else None
    grouped = {field: {value: {"inputs": len(selected := [row for row in records if row[field] == value]),
        "candidates": sum(row["selection"] == "candidate" for row in selected)}
        for value in sorted({row[field] for row in records})} for field in ("wording", "category", "family_id")}
    candidate_text = "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in candidates)
    candidate_path = output / "candidate_preferences.jsonl"
    if args.audit_only:
        if candidate_path.read_text(encoding="utf-8") != candidate_text:
            raise ValueError("candidate preferences differ from mined outputs")
    else:
        candidate_path.write_text(candidate_text, encoding="utf-8", newline="\n")
    summary = {**inventory, "training_inputs": len(records), "strict_pass": sum(row["end_to_end_pass"] for row in records),
        "semantic_pass": sum(row["semantic_pass"] for row in records), "candidates": len(candidates),
        "candidate_families": len({row.family_id for row in candidates}),
        "selection_counts": dict(Counter(row["selection"] for row in records)),
        "candidate_failures": dict(Counter(row.provenance["rejected_failure"] for row in candidates)),
        "by_group": grouped, "audit": audit,
        "files": {name: file_hash(output / name) for name in ("sft_train.jsonl", "candidate_preferences.jsonl")},
        "training_ready": False, "baseline_replaced": False,
        "limitations": ["仅采集训练输入，不推断泛化收益", "只保留错误会改变题型与family分布，尚未冻结新训练设计",
            "单次贪心输出，可能漏掉其他错误", "问候评分仅检查结构与路由，不验证语气和答案含义"]}
    if args.audit_only:
        if summary != json.loads((output / "summary.json").read_text(encoding="utf-8")):
            raise ValueError("saved summary differs")
    else:
        write_json(output / "summary.json", summary)
        lines = ["# 冻结SFT训练错误采集", "", f"训练输入{len(records)}条，严格通过{summary['strict_pass']}，语义通过{summary['semantic_pass']}；可审计候选{len(candidates)}对，覆盖{summary['candidate_families']}个family。",
            "", "未生成开发题回答，未训练；候选尚非冻结的新训练集。", "", "| 措辞 | 输入 | 候选 |", "| --- | --- | --- |"]
        for key, values in grouped["wording"].items():
            lines.append(f"| {key} | {values['inputs']} | {values['candidates']} |")
        lines += ["", "## 限制", "", *[f"- {item}" for item in summary["limitations"]]]
        (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "candidates": len(candidates), "audit_consistent": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
