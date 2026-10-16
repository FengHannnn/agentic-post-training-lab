"""Audit the three v2 DPO runs and summarize fixed-answer scores, not generation."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from agentic_post_training.data.preferences import load_preference_splits

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"small_28": ("train_small.jsonl", 28, 28),
        "small_168": ("train_small.jsonl", 168, 28),
        "large_168": ("train.jsonl", 168, 168)}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="核对三分支规模对照并汇总固定答案评分，不生成回答")
    parser.add_argument("--runs-file", type=Path, required=True)
    args = parser.parse_args()
    runs = json.loads(args.runs_file.read_text(encoding="utf-8"))
    if set(runs) != set(ARMS):
        raise ValueError("all three planned arms are required")
    summaries, histories = {}, {}
    for arm, (filename, steps, count) in ARMS.items():
        folder = Path(runs[arm])
        result = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        config = result["config"]
        if config["train_file"] != filename or config["max_steps"] != steps or result["steps_completed"] != steps:
            raise ValueError(f"{arm}: wrong training split or steps")
        samples = load_preference_splits(ROOT / config["preference_data"], result["data_manifest"], filename)
        if len(samples["train"]) != count or len(samples["development"]) != 56:
            raise ValueError(f"{arm}: unexpected data counts")
        if any(not result[key] for key in ("reference_unchanged", "reference_no_grad", "policy_frozen_parameters_unchanged",
                    "policy_updated", "initial_policy_reference_match", "reload_logits_match")):
            raise ValueError(f"{arm}: training invariant failed")
        history = [json.loads(line) for line in (folder / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(history) != steps or [row["step"] for row in history] != list(range(1, steps + 1)):
            raise ValueError(f"{arm}: incomplete history")
        for row in history:
            if samples["train"][row["pair_index"]].id != row["pair_id"]:
                raise ValueError(f"{arm}: sample index/ID mismatch")
        expected_ids = {row.id for row in samples["train"]}
        for start in range(0, steps, count):
            epoch = history[start:start + count]
            if len(epoch) != count or {row["pair_id"] for row in epoch} != expected_ids:
                raise ValueError(f"{arm}: incomplete or repeated samples within epoch")
        exposures = Counter(row["pair_id"] for row in history)
        if set(exposures.values()) != {steps // count}:
            raise ValueError(f"{arm}: incorrect exposures")
        for key, value in result["training_token_counts"].items():
            if value != sum(row[key] for row in history):
                raise ValueError(f"{arm}: token accounting mismatch")
        summaries[arm] = result
        histories[arm] = history
    first = summaries["small_28"]
    varying = {"train_file", "max_steps", "experiment_id"}
    fixed_config = {key: value for key, value in first["config"].items() if key not in varying}
    for arm, result in summaries.items():
        if {key: value for key, value in result["config"].items() if key not in varying} != fixed_config:
            raise ValueError(f"{arm}: non-controlled configuration difference")
        for key in ("model", "resolved_revision", "baseline_manifest_sha256", "data_manifest", "source_sha256"):
            if result[key] != first[key]:
                raise ValueError(f"{arm}: differing model/data/source identity")
        if result["initial_policy_sha256"] != first["initial_policy_sha256"] or result["initial_reference_sha256"] != first["initial_policy_sha256"]:
            raise ValueError(f"{arm}: differing initial parameters")
        for key, value in result["before"]["development"].items():
            if not math.isclose(value, first["before"]["development"][key], abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError(f"{arm}: initial development scores differ")
    for left, right in zip(histories["small_28"], histories["small_168"][:28]):
        if left["pair_id"] != right["pair_id"]:
            raise ValueError("small-run prefix order differs")
        for key in ("loss", "relative_margin", "chosen_log_prob", "rejected_log_prob", "grad_norm"):
            if not math.isclose(left[key], right[key], abs_tol=1e-5, rel_tol=1e-5):
                raise ValueError(f"small-run prefix score differs: {key}")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/evaluations" / f"dpo-v2-scaling-training-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    result = {"runs": runs, "all_controls_verified": True, "small_28_matches_small_168_prefix": True,
        "summary_sha256": {arm: file_hash(Path(path) / "summary.json") for arm, path in runs.items()},
        "adapter_sha256": {arm: file_hash(Path(path) / "adapter/adapter_model.safetensors") for arm, path in runs.items()},
        "results": summaries, "metric_scope": "fixed-answer preference scores; no generation or behavior results"}
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    lines = ["# v2 DPO 规模对照：训练与固定答案评分", "", "本轮不生成回答，relative preference accuracy不是生成正确率。", "",
        "| 分支 | 训练对数 | 更新次数 | 训练后loss | 开发后loss | 开发相对偏好改善 | 开发chosen Δ | 开发rejected Δ | 更新耗时/秒 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm, (_, steps, count) in ARMS.items():
        row = summaries[arm]
        before, after = row["before"]["development"], row["after"]["development"]
        passed = round(after["preference_accuracy"] * 56)
        lines.append(f"| {arm} | {count} | {steps} | {row['after']['train']['loss']:.6f} | {after['loss']:.6f} | {passed}/56 | {after['chosen_log_prob']-before['chosen_log_prob']:+.6f} | {after['rejected_log_prob']-before['rejected_log_prob']:+.6f} | {row['training_seconds']:.2f} |")
    lines += ["", "控制核对通过：固定配置、基线、初始参数、代码和数据身份一致；开发初始评分一致；A/B前28步的样本顺序及数值指标一致。reference冻结无梯度，policy仅LoRA更新，三组保存重载检查通过。", "",
        "| 分支 | chosen监督token | rejected监督token | 非padding输入token | padding后输入token |",
        "| --- | --- | --- | --- | --- |"]
    for arm in ARMS:
        counts = summaries[arm]["training_token_counts"]
        lines.append("| " + " | ".join([arm] + [str(counts[key]) for key in ("chosen_response_tokens", "rejected_response_tokens", "input_tokens", "padded_input_tokens")]) + " |")
    lines += ["", "token与耗时只统计policy训练循环，未计reference缓存、训练前后评分、模型加载/保存和重载；固定步数不等于完整运行计算量完全相同。", "",
        "A→B改变同一小集的重复更新，B→C在相同总曝光下改变覆盖。单种子、单次小子集、合成负例与共享模板限制结论；尚未进行新开发集基线生成和三分支逐题行为对照。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "runs": runs}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
