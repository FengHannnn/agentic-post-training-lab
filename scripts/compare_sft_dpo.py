"""Generate fresh SFT/DPO behavior under identical settings and compare by ID."""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from freeze_sft_baseline import verify
from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, semantic_pass
from agentic_post_training.data.validation import target_for_example
from agentic_post_training.environments.calculator import expression_structure
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import load_eval_cases, score_output

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def table_text(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="同条件比较冻结 v2 SFT 与 DPO 的逐题行为")
    parser.add_argument("--dpo-run", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    run = args.dpo_run.resolve()
    training = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    baseline = ROOT / training["config"]["baseline"]
    verify(baseline)
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(baseline / "manifest.json") != training["baseline_manifest_sha256"]:
        raise ValueError("DPO baseline identity mismatch")
    if (training["model"], training["resolved_revision"]) != (manifest["model"], manifest["resolved_revision"]):
        raise ValueError("SFT and DPO must share the exact base")
    tokenizers = {label: HFChatTokenizer(AutoTokenizer.from_pretrained(str(path), local_files_only=True,
                    trust_remote_code=False)) for label, path in (
                    ("sft", baseline / "checkpoint/tokenizer"), ("dpo", run / "tokenizer"))}
    left, right = tokenizers["sft"], tokenizers["dpo"]
    if (left.raw.get_vocab() != right.raw.get_vocab() or left.raw.chat_template != right.raw.chat_template
        or left.pad_token_id != right.pad_token_id or left.raw.eos_token_id != right.raw.eos_token_id
        or left.raw.special_tokens_map != right.raw.special_tokens_map):
        raise ValueError("tokenizers/templates/special tokens differ")
    paths = {name: baseline / "project/data/eval" / filename for name, filename in (
        ("tool_calls", "tool_calls_v1.jsonl"), ("brackets", "bracket_diagnostic_v1.jsonl"),
        ("spacing_control", "bracket_spacing_control_v1.jsonl"))}
    suites = {name: load_eval_cases(path) for name, path in paths.items()}
    dev_path = ROOT / training["config"]["preference_data"] / "development.jsonl"
    if file_hash(dev_path) != training["data_manifest"]["files"]["development.jsonl"]:
        raise ValueError("preference development data changed since training")
    suites["preference_development"] = [row.target() for row in load_preferences(dev_path)]
    paths["preference_development"] = dev_path
    sft_targets = [target_for_example(row) for row in load_examples(
        baseline / "project" / manifest["training_summary"]["config"]["train_data"])]
    dpo_train_path = ROOT / training["config"]["preference_data"] / "train.jsonl"
    if file_hash(dpo_train_path) != training["data_manifest"]["files"]["train.jsonl"]:
        raise ValueError("preference training data changed")
    dpo_targets = [row.target() for row in load_preferences(dpo_train_path)]
    seen = {}
    for label, targets in (("sft", sft_targets), ("dpo", dpo_targets)):
        seen[label] = ({(case.system, case.prompt) for case in targets},
                       {expression_structure(case.expected_expression) for case in targets
                        if case.expected_type == "tool_call"})
    settings = manifest["generation"]
    set_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    print(f"加载固定 base 和两个 adapter；每份生成 {sum(len(cases) for cases in suites.values())} 题，不更新权重。", flush=True)
    base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(baseline / "checkpoint/adapter"), adapter_name="sft", is_trainable=False)
    model.load_adapter(str(run / "adapter"), adapter_name="dpo", is_trainable=False)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/evaluations" / f"sft-dpo-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    records = {label: {} for label in ("sft", "dpo")}
    for label in records:
        model.set_adapter(label)
        model.requires_grad_(False)
        for suite, cases in suites.items():
            records[label][suite] = []
            with (output / f"{label}_{suite}.jsonl").open("w", encoding="utf-8", newline="\n") as log:
                for index, case in enumerate(cases, 1):
                    messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
                    if left.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) != right.apply_chat_template(messages, tokenize=True, add_generation_prompt=True):
                        raise ValueError(f"{case.id}: tokenized prompts differ")
                    generation = generate_prediction(model, tokenizers[label], case, settings["max_new_tokens"])
                    score = score_output(generation["prediction"], case)
                    record = {**asdict(case), **generation, **score,
                              "semantic_pass": semantic_pass(score, case.expected_type)}
                    for source, (prompts, expressions) in seen.items():
                        record[f"seen_{source}_training_prompt"] = (case.system, case.prompt) in prompts
                        record[f"seen_{source}_training_expression"] = case.expected_type == "tool_call" and expression_structure(case.expected_expression) in expressions
                    if suite in ("brackets", "spacing_control"):
                        number_set, wording, form = case.category.split("/")
                        record.update(number_set=number_set, wording=wording, form=form,
                                      pair_id=f"{number_set}/{wording}/{form.split('_')[0]}")
                    records[label][suite].append(record)
                    log.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log.flush()
                    print(f"{label} {suite} {index}/{len(cases)} {case.id}: {score['failure'] or 'pass'}", flush=True)
    comparisons = {}
    historical = {}
    report_date = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    lines = ["# SFT-only 与 SFT＋DPO 逐项行为对比", "", f"日期：{report_date}。所有数据为已查看开发诊断；包含已见 SFT 表达式探针，没有最终未查看测试。", "", "| 数据 | SFT 严格 | DPO 严格 | SFT 数值/路由 | DPO 数值/路由 | 严格改善/退步 |", "| --- | --- | --- | --- | --- | --- |"]
    changes_by_suite = {}
    for suite in suites:
        result, changes = compare_records(records["sft"][suite], records["dpo"][suite],
                                          paired=suite in ("brackets", "spacing_control"))
        comparisons[suite] = result
        changes_by_suite[suite] = changes
        write_json(output / f"{suite}_changes.json", changes)
        if suite != "preference_development":
            saved_file = baseline / "evaluations" / suite / ("lora_predictions.jsonl" if suite == "tool_calls" else "v2_predictions.jsonl")
            saved = {row["id"]: row for row in (json.loads(line) for line in saved_file.read_text(encoding="utf-8").splitlines())}
            if set(saved) != {row["id"] for row in records["sft"][suite]}:
                raise ValueError("historical baseline IDs differ")
            differing = [row["id"] for row in records["sft"][suite] if row["prediction"] != saved[row["id"]]["prediction"]]
            historical[suite] = {"total": len(saved), "prediction_mismatch_ids": differing}
        cells = []
        for label in ("sft", "dpo"):
            metric = result[label]["metrics"]["end_to_end_pass"]
            cells.append(f"{metric['passed']}/{metric['total']}")
        for label in ("sft", "dpo"):
            metric = result[label]["semantic_pass"]
            cells.append(f"{metric['passed']}/{metric['total']}")
        delta = result["transitions"]["end_to_end_pass"]
        lines.append(f"| {suite} | {' | '.join(cells)} | {delta['improved']}/{delta['regressed']} |")
    for suite, changes in changes_by_suite.items():
        lines += ["", f"## {suite} 逐题输出", "", "| ID | 题目 | 期望表达式 | SFT 输出 | DPO 输出 | 严格变化 | 数值/路由变化 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for row in changes:
            lines.append("| " + " | ".join(table_text(row[key]) for key in (
                "id", "prompt", "expected_expression", "sft_prediction", "dpo_prediction", "end_to_end_pass", "semantic_pass")) + " |")
    result = {"sft_baseline": str(baseline), "dpo_run": str(run), "model": manifest["model"],
              "resolved_revision": manifest["resolved_revision"], "generation": settings,
              "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
              "dpo_summary_sha256": file_hash(run / "summary.json"),
              "dpo_adapter_sha256": file_hash(run / "adapter/adapter_model.safetensors"),
              "evaluation_data_sha256": {name: file_hash(path) for name, path in paths.items()},
              "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
              "historical_sft_reproduction": historical, "suites": comparisons}
    write_json(output / "summary.json", result)
    lines += ["", "历史 SFT 重现情况（预测文本差异）：", "", "```json", json.dumps(historical, ensure_ascii=False, indent=2), "```", "", "严格指标保留运算 AST；数值指标只验证当前数字，不证明符号等价。问候只评分结构与路由，不评分回答语义。不同数据组含重叠问题，不将总计当独立测试样本准确率。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "results": {name: {
        "sft": data["sft"]["metrics"]["end_to_end_pass"], "dpo": data["dpo"]["metrics"]["end_to_end_pass"],
        "changes": data["transitions"]["end_to_end_pass"]} for name, data in comparisons.items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
