"""Compare chosen-only continued SFT with pinned SFT/DPO behavior records."""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, table_text, write_json
from freeze_sft_baseline import verify
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, semantic_pass
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import EvalCase, load_eval_cases, score_output

ROOT = Path(__file__).resolve().parents[1]


def load_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="继续 SFT 与原 SFT、DPO 的逐题对照")
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--behavior-run", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    run, behavior = args.control_run.resolve(), args.behavior_run.resolve()
    training = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    previous = json.loads((behavior / "summary.json").read_text(encoding="utf-8"))
    dpo_run = Path(training["dpo_run"])
    baseline = ROOT / training["config"]["baseline"]
    verify(baseline)
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if (file_hash(baseline / "manifest.json") != training["baseline_manifest_sha256"]
        or previous["baseline_manifest_sha256"] != training["baseline_manifest_sha256"]
        or file_hash(dpo_run / "summary.json") != previous["dpo_summary_sha256"]
        or file_hash(dpo_run / "adapter/adapter_model.safetensors") != previous["dpo_adapter_sha256"]
        or (training["model"], training["resolved_revision"]) != (previous["model"], previous["resolved_revision"])
        or previous["generation"] != manifest["generation"]):
        raise ValueError("control and historical experiment identities differ")
    paths = {name: baseline / "project/data/eval" / filename for name, filename in (
        ("tool_calls", "tool_calls_v1.jsonl"), ("brackets", "bracket_diagnostic_v1.jsonl"),
        ("spacing_control", "bracket_spacing_control_v1.jsonl"))}
    paths["preference_development"] = ROOT / training["config"]["preference_data"] / "development.jsonl"
    for name, path in paths.items():
        if file_hash(path) != previous["evaluation_data_sha256"][name]:
            raise ValueError(f"evaluation data changed: {name}")
    tokenizers = [HFChatTokenizer(AutoTokenizer.from_pretrained(str(path), local_files_only=True,
        trust_remote_code=False)) for path in (baseline / "checkpoint/tokenizer", dpo_run / "tokenizer", run / "tokenizer")]
    tokenizer = tokenizers[0]
    for candidate in tokenizers[1:]:
        if (tokenizer.raw.get_vocab() != candidate.raw.get_vocab()
            or tokenizer.raw.chat_template != candidate.raw.chat_template
            or tokenizer.raw.special_tokens_map != candidate.raw.special_tokens_map
            or tokenizer.pad_token_id != candidate.pad_token_id
            or tokenizer.raw.eos_token_id != candidate.raw.eos_token_id):
            raise ValueError("tokenizer identity differs")
    settings = previous["generation"]
    set_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    base = AutoModelForCausalLM.from_pretrained(training["model"], revision=training["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(run / "adapter"), is_trainable=False)
    model.requires_grad_(False)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/evaluations" / f"sft-control-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    results, source_hashes = {}, {}
    lines = ["# chosen-only 继续 SFT 对照", "", "原 SFT/DPO 复用已核对身份与设置的历史输出，继续 SFT 新生成90题；均为开发诊断。", "",
        "| 组别 | 原 SFT | DPO | 继续 SFT | 相对原 SFT 改善/退步 | 相对 DPO 改善/退步 |",
        "| --- | --- | --- | --- | --- | --- |"]
    detail_lines = []
    for suite in previous["suites"]:
        source = {label: load_records(behavior / f"{label}_{suite}.jsonl") for label in ("sft", "dpo")}
        source_hashes[suite] = {label: file_hash(behavior / f"{label}_{suite}.jsonl") for label in source}
        # Recompute the historical summary instead of trusting a possibly edited JSONL.
        historical, _ = compare_records(source["sft"], source["dpo"], paired=suite in ("brackets", "spacing_control"))
        if historical != previous["suites"][suite]:
            raise ValueError(f"historical records/summary differ: {suite}")
        expected_cases = ([row.target() for row in load_preferences(paths[suite])] if suite == "preference_development"
                          else load_eval_cases(paths[suite]))
        expected = {case.id: case for case in expected_cases}
        if set(expected) != {row["id"] for row in source["sft"]}:
            raise ValueError("historical evaluation IDs differ")
        for label, records in source.items():
            for row in records:
                if any(row[key] != getattr(expected[row["id"]], key) for key in
                       ("system", "prompt", "category", "expected_type", "expected_expression", "expected_result")):
                    raise ValueError("historical input/target differs from pinned data")
        controlled = []
        with (output / f"control_{suite}.jsonl").open("w", encoding="utf-8", newline="\n") as log:
            for index, left in enumerate(source["sft"], 1):
                case = EvalCase(**{key: left[key] for key in ("id", "category", "prompt", "expected_type", "expected_expression", "expected_result", "system")})
                messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
                if any(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) != candidate.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) for candidate in tokenizers[1:]):
                    raise ValueError("prompt token IDs differ")
                generation = generate_prediction(model, tokenizer, case, settings["max_new_tokens"])
                score = score_output(generation["prediction"], case)
                row = {**left, **generation, **score, "semantic_pass": semantic_pass(score, case.expected_type)}
                controlled.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                print(f"control {suite} {index}/{len(source['sft'])}: {score['failure'] or 'pass'}", flush=True)
        comparisons = {}
        for label in source:
            result, changes = compare_records(source[label], controlled, paired=suite in ("brackets", "spacing_control"))
            comparisons[label] = {"baseline": result["sft"], "control": result["dpo"],
                                  **{key: value for key, value in result.items() if key not in ("sft", "dpo")}}
            changes = [{("baseline_" + key[4:] if key.startswith("sft_") else
                         "control_" + key[4:] if key.startswith("dpo_") else key): value
                        for key, value in row.items()} for row in changes]
            write_json(output / f"{suite}_{label}_vs_control_changes.json", changes)
        results[suite] = comparisons
        def cell(label):
            comparison = comparisons["dpo" if label == "dpo" else "sft"]
            metric = comparison["control" if label == "control" else "baseline"]["metrics"]["end_to_end_pass"]
            return f"{metric['passed']}/{metric['total']}"
        transitions = [comparisons[label]["transitions"]["end_to_end_pass"] for label in source]
        lines.append(f"| {suite} | {cell('sft')} | {cell('dpo')} | {cell('control')} | {transitions[0]['improved']}/{transitions[0]['regressed']} | {transitions[1]['improved']}/{transitions[1]['regressed']} |")
        detail_lines += ["", f"## {suite} 逐题输出", "", "| ID | 题目 | SFT | DPO | 继续 SFT |", "| --- | --- | --- | --- | --- |"]
        dpo_by_id = {row["id"]: row for row in source["dpo"]}
        for left, right in zip(source["sft"], controlled):
            detail_lines.append("| " + " | ".join(table_text(value) for value in (left["id"], left["prompt"], left["prediction"], dpo_by_id[left["id"]]["prediction"], right["prediction"])) + " |")
    result = {"control_run": str(run), "behavior_run": str(behavior), "generation": settings,
        "baseline_manifest_sha256": training["baseline_manifest_sha256"],
        "control_adapter_sha256": file_hash(run / "adapter/adapter_model.safetensors"),
        "control_summary_sha256": file_hash(run / "summary.json"), "historical_record_sha256": source_hashes,
        "comparison_key_convention": "outer sft/dpo indicates baseline; inner baseline=that baseline, control=continued SFT",
        "suites": results}
    write_json(output / "summary.json", result)
    lines += ["", "训练步数与 chosen 暴露次数相同，计算量、训练目标和梯度尺度不同；单种子小数据不能证明方法普遍优劣。不同组存在重叠题目，不合并为独立测试准确率。"]
    (output / "report.md").write_text("\n".join(lines + detail_lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
