"""Score fixed preference answers before/after DPO; no generation or updates."""

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from freeze_sft_baseline import verify
from build_dpo_preferences import protected_cases
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, validate_preferences
from agentic_post_training.evaluation.preference_diagnostics import (
    grouped_preference_changes, summarize_preference_changes,
)
from agentic_post_training.trainers.dpo import encode_pair, objective, pair_log_probs

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="分组诊断 DPO 的 chosen/rejected 概率变化")
    parser.add_argument("--dpo-run", type=Path, required=True)
    parser.add_argument("--behavior-run", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    run, behavior = args.dpo_run.resolve(), args.behavior_run.resolve()
    training = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    behavior_summary = json.loads((behavior / "summary.json").read_text(encoding="utf-8"))
    if (file_hash(run / "adapter/adapter_model.safetensors") != behavior_summary["dpo_adapter_sha256"]
        or file_hash(run / "summary.json") != behavior_summary["dpo_summary_sha256"]):
        raise ValueError("behavior results and DPO checkpoint differ")
    baseline = ROOT / training["config"]["baseline"]
    verify(baseline)
    if file_hash(baseline / "manifest.json") != training["baseline_manifest_sha256"]:
        raise ValueError("baseline identity mismatch")
    folder = ROOT / training["config"]["preference_data"]
    for name, digest in training["data_manifest"]["files"].items():
        if file_hash(folder / name) != digest:
            raise ValueError(f"preference data changed: {name}")
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    samples = {split: load_preferences(folder / f"{split}.jsonl") for split in ("train", "development")}
    validate_preferences(samples["train"], samples["development"], protected_cases(), tokenizer,
                         training["config"]["max_length"])
    pairs = {split: [encode_pair(row, tokenizer, training["config"]["max_length"]) for row in rows]
             for split, rows in samples.items()}
    set_seed(training["config"]["seed"])
    torch.set_num_threads(training["config"]["cpu_threads"])
    base = AutoModelForCausalLM.from_pretrained(training["model"], revision=training["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(baseline / "checkpoint/adapter"), adapter_name="sft", is_trainable=False)
    model.load_adapter(str(run / "adapter"), adapter_name="dpo", is_trainable=False)
    scores = {label: {} for label in ("sft", "dpo")}
    for label in scores:
        model.set_adapter(label)
        model.requires_grad_(False)
        model.eval()
        for split, selected in pairs.items():
            scores[label][split] = []
            for index, pair in enumerate(selected, 1):
                scores[label][split].append(pair_log_probs(model, pair, tokenizer.pad_token_id).detach().clone())
                print(f"{label} {split} {index}/{len(selected)}", flush=True)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/evaluations" / f"dpo-preference-diagnostic-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    dev_changes = {row["id"]: row for row in json.loads((behavior / "preference_development_changes.json").read_text(encoding="utf-8"))}
    results = {}
    all_rows = []
    for split, selected in samples.items():
        rows = []
        for index, example in enumerate(selected):
            sft, dpo = scores["sft"][split][index], scores["dpo"][split][index]
            loss = objective(dpo, sft, training["config"]["beta"])
            row = {"id": example.id, "split": split, "category": example.category,
                   "expected_type": example.expected_type, "rejected_source": example.provenance["rejected_source"],
                   "prompt": example.prompt, "chosen": example.chosen, "rejected": example.rejected,
                   "chosen_tokens": sum(pairs[split][index][0]["loss_mask"]),
                   "rejected_tokens": sum(pairs[split][index][1]["loss_mask"]),
                   "sft_chosen_log_prob": sft[0].item(), "sft_rejected_log_prob": sft[1].item(),
                   "dpo_chosen_log_prob": dpo[0].item(), "dpo_rejected_log_prob": dpo[1].item(),
                   "delta_chosen": (dpo[0] - sft[0]).item(), "delta_rejected": (dpo[1] - sft[1]).item(),
                   "relative_margin": loss.logits.item(), "loss": loss.loss.item(),
                   "generation_transition": dev_changes[example.id]["end_to_end_pass"] if split == "development" else None}
            rows.append(row)
        overall = summarize_preference_changes(rows)
        # Reproduce the previous run's aggregate metrics before interpreting groups.
        for key, diagnostic_key in (("loss", "loss"), ("relative_margin", "relative_margin"),
                                     ("chosen_log_prob", "dpo_chosen_log_prob"),
                                     ("rejected_log_prob", "dpo_rejected_log_prob")):
            if not math.isclose(overall[diagnostic_key], training["after"][split][key], abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError(f"{split}: stored training metric mismatch: {key}")
        results[split] = {"overall": overall, **{f"by_{field}": grouped_preference_changes(rows, field)
                          for field in ("category", "rejected_source", "expected_type")}}
        all_rows += rows
    (output / "pair_scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8", newline="\n")
    result = {"dpo_run": str(run), "behavior_run": str(behavior),
              "dpo_adapter_sha256": file_hash(run / "adapter/adapter_model.safetensors"),
              "baseline_manifest_sha256": training["baseline_manifest_sha256"],
              "data_files": training["data_manifest"]["files"], "beta": training["config"]["beta"],
              "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
              "reproduces_training_after_metrics": True, "splits": results}
    write_json(output / "summary.json", result)
    lines = ["# DPO 偏好概率分组诊断", "", "不训练、不生成，仅评分固定的 chosen/rejected。正 delta 表示序列 log probability 增加。", "", "| split/类别 | 对数 | chosen Δ均值 | rejected Δ均值 | 相对改善 | chosen下降但相对改善 |", "| --- | --- | --- | --- | --- | --- |"]
    for split, groups in results.items():
        for category, values in {"overall": groups["overall"], **groups["by_category"]}.items():
            lines.append(f"| {split}/{category} | {values['pairs']} | {values['delta_chosen']:.6f} | {values['delta_rejected']:.6f} | {values['relative_improved']} | {values['chosen_decreased_but_relative_improved']} |")
    lines += ["", "来源与题型混杂：真实错误负例多集中在无括号自然语言题，合成负例多集中在括号/问候题，不能把分组差异直接归因于来源。", "", "逐条数据见 pair_scores.jsonl；开发题实际生成变化来自已完成的行为比较，不把训练题评分当泛化结果。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "overall": {split: groups["overall"]
                     for split, groups in results.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
