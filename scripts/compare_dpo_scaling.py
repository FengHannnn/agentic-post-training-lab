"""Greedy behavior comparison of frozen SFT and three audited DPO scaling runs."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, table_text, write_json
from freeze_sft_baseline import verify
from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, load_preference_splits
from agentic_post_training.data.validation import target_for_example
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.scaling import GENERATION_FIELDS, behavior_record, training_keys, v2_subgroups
from agentic_post_training.evaluation.tool_calls import load_eval_cases

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("sft", "small_28", "small_168", "large_168")


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="SFT与v2 DPO三分支逐题行为对照，不训练")
    parser.add_argument("--training-comparison", type=Path, required=True)
    parser.add_argument("--historical-behavior", type=Path,
        default=ROOT / "artifacts/evaluations/sft-dpo-20261010-150134-fc7e5e9a")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output is None:
        raise ValueError("resume requires an explicit output directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    training_path = args.training_comparison.resolve()
    training = json.loads((training_path / "summary.json").read_text(encoding="utf-8"))
    if not training["all_controls_verified"] or set(training["runs"]) != set(LABELS[1:]):
        raise ValueError("three verified training runs are required")
    first = training["results"]["small_28"]
    baseline = ROOT / first["config"]["baseline"]
    verify(baseline)
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if file_hash(baseline / "manifest.json") != first["baseline_manifest_sha256"]:
        raise ValueError("baseline changed")
    selected, adapter_paths, tokenizer_paths = {}, {"sft": baseline / "checkpoint/adapter"}, {"sft": baseline / "checkpoint/tokenizer"}
    for label, path in training["runs"].items():
        run = Path(path)
        if (file_hash(run / "summary.json") != training["summary_sha256"][label]
            or file_hash(run / "adapter/adapter_model.safetensors") != training["adapter_sha256"][label]):
            raise ValueError(f"{label}: training artifact changed")
        result = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        if result != training["results"][label] or result["baseline_manifest_sha256"] != first["baseline_manifest_sha256"]:
            raise ValueError("training summary/baseline differs")
        selected[label] = load_preference_splits(ROOT / result["config"]["preference_data"],
            result["data_manifest"], result["config"].get("train_file", "train.jsonl"))
        adapter_paths[label], tokenizer_paths[label] = run / "adapter", run / "tokenizer"
    tokenizers = {label: HFChatTokenizer(AutoTokenizer.from_pretrained(str(path), local_files_only=True,
        trust_remote_code=False)) for label, path in tokenizer_paths.items()}
    tokenizer = tokenizers["sft"]
    for other in tokenizers.values():
        if (tokenizer.raw.get_vocab() != other.raw.get_vocab() or tokenizer.raw.chat_template != other.raw.chat_template
            or tokenizer.raw.special_tokens_map != other.raw.special_tokens_map
            or tokenizer.pad_token_id != other.pad_token_id or tokenizer.raw.eos_token_id != other.raw.eos_token_id):
            raise ValueError("tokenizers/templates/special tokens differ")
    preference_rows = selected["small_28"]["development"]
    if any(rows["development"] != preference_rows for rows in selected.values()):
        raise ValueError("development sets differ")
    preference_by_id = {row.id: row for row in preference_rows}
    paths = {"tool_calls": baseline / "project/data/eval/tool_calls_v1.jsonl",
             "brackets": baseline / "project/data/eval/bracket_diagnostic_v1.jsonl",
             "spacing_control": baseline / "project/data/eval/bracket_spacing_control_v1.jsonl",
             "preference_v1_development": ROOT / "data/preferences/dpo_v1/development.jsonl",
             "preference_v2_development": ROOT / first["config"]["preference_data"] / "development.jsonl"}
    suites = {"preference_v2_development": [row.target() for row in preference_rows]}
    for name in ("tool_calls", "brackets", "spacing_control"):
        suites[name] = load_eval_cases(paths[name])
    suites["preference_v1_development"] = [row.target() for row in load_preferences(paths["preference_v1_development"])]
    sft_cases = [target_for_example(row) for row in load_examples(
        baseline / "project" / manifest["training_summary"]["config"]["train_data"])]
    seen = {"sft": training_keys(sft_cases), **{label: training_keys([row.target() for row in data["train"]])
                                            for label, data in selected.items()}}
    settings = manifest["generation"]
    historical_path = args.historical_behavior.resolve()
    old = json.loads((historical_path / "summary.json").read_text(encoding="utf-8"))
    if (old["baseline_manifest_sha256"] != first["baseline_manifest_sha256"]
        or old["generation"] != settings or (old["model"], old["resolved_revision"]) != (manifest["model"], manifest["resolved_revision"])):
        raise ValueError("historical baseline/settings differ")
    if any(version(name) != value for name, value in old["versions"].items()):
        raise ValueError("historical runtime library versions differ")
    generation_source = ROOT / "src/agentic_post_training/evaluation/generation.py"
    if generation_source.read_text(encoding="utf-8") != (baseline / "project/src/agentic_post_training/evaluation/generation.py").read_text(encoding="utf-8"):
        raise ValueError("generation implementation differs from frozen baseline")
    historical, historical_hashes = {}, {}
    for suite in ("tool_calls", "brackets", "spacing_control", "preference_v1_development"):
        old_name = "preference_development" if suite == "preference_v1_development" else suite
        if file_hash(paths[suite]) != old["evaluation_data_sha256"][old_name]:
            raise ValueError("historical evaluation data changed")
        path = historical_path / f"sft_{old_name}.jsonl"
        historical_hashes[suite] = file_hash(path)
        records = load_jsonl(path)
        if len(records) != len(suites[suite]) or {row["id"] for row in records} != {case.id for case in suites[suite]}:
            raise ValueError("historical IDs differ or repeat")
        indexed = {row["id"]: row for row in records}
        historical[suite] = []
        for case in suites[suite]:
            saved = indexed[case.id]
            if any(saved[key] != value for key, value in asdict(case).items()):
                raise ValueError("historical inputs/targets differ")
            row = behavior_record(case, saved, seen, legacy_paired=suite in ("brackets", "spacing_control"), source="historical_sft")
            for key in ("end_to_end_pass", "semantic_pass", "json_valid", "schema_valid", "route_correct", "failure"):
                if row[key] != saved[key]:
                    raise ValueError("historical scores do not reproduce")
            historical[suite].append(row)
    historical_cache = {}
    for group in historical.values():
        for row in group:
            key = (row["system"], row["prompt"])
            if key in historical_cache and any(row[field] != historical_cache[key][field] for field in GENERATION_FIELDS):
                raise ValueError("same historical input has inconsistent generation records")
            historical_cache[key] = row
    inventory = {"training_summary_sha256": file_hash(training_path / "summary.json"),
        "baseline_manifest_sha256": first["baseline_manifest_sha256"], "generation": settings,
        "adapter_sha256": {label: file_hash(path / "adapter_model.safetensors") for label, path in adapter_paths.items()},
        "evaluation_sha256": {name: file_hash(path) for name, path in paths.items()}, "historical_records_sha256": historical_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
            (Path(__file__), generation_source, ROOT / "src/agentic_post_training/evaluation/scaling.py",
             ROOT / "src/agentic_post_training/evaluation/tool_calls.py", ROOT / "src/agentic_post_training/evaluation/comparison.py",
             ROOT / "src/agentic_post_training/evaluation/brackets.py")},
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")}}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else ROOT / "artifacts/evaluations" / f"dpo-v2-scaling-behavior-{stamp}-{uuid4().hex[:8]}"
    if args.resume:
        if json.loads((output / "run_manifest.json").read_text(encoding="utf-8")) != inventory:
            raise ValueError("resume identity/settings/source differ")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", inventory)
    print(json.dumps({"output": str(output), "records_per_model": sum(map(len, suites.values()))}, ensure_ascii=False), flush=True)
    set_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(adapter_paths["sft"]), adapter_name="sft", is_trainable=False)
    for label in LABELS[1:]:
        model.load_adapter(str(adapter_paths[label]), adapter_name=label, is_trainable=False)
    model.set_adapter("sft")
    model.requires_grad_(False)
    probes = []
    for suite in ("tool_calls", "brackets", "spacing_control"):
        case = suites[suite][0]
        prediction = generate_prediction(model, tokenizer, case, settings["max_new_tokens"])
        if any(prediction[field] != historical[suite][0][field] for field in GENERATION_FIELDS):
            raise ValueError("historical SFT reuse probe did not reproduce")
        probes.append({"id": case.id, "prediction_matches": True})
    records = {label: {} for label in LABELS}
    for label in LABELS:
        model.set_adapter(label)
        model.requires_grad_(False)
        cache = dict(historical_cache) if label == "sft" else {}
        for suite, cases in suites.items():
            path = output / f"{label}_{suite}.jsonl"
            done = load_jsonl(path) if args.resume and path.exists() else []
            if len(done) > len(cases) or [row["id"] for row in done] != [case.id for case in cases[:len(done)]]:
                raise ValueError("resume records must be a unique ordered prefix")
            records[label][suite] = []
            with path.open("a" if done else "w", encoding="utf-8", newline="\n") as log:
                for index, case in enumerate(cases, 1):
                    messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
                    if any(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) != other.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) for other in tokenizers.values()):
                        raise ValueError("tokenized prompts differ")
                    key = (case.system, case.prompt)
                    preference = preference_by_id.get(case.id) if suite == "preference_v2_development" else None
                    if index <= len(done):
                        old_row = done[index - 1]
                        row = behavior_record(case, old_row, seen, preference=preference,
                            legacy_paired=suite in ("brackets", "spacing_control"), source=old_row["generation_source"])
                        if row != old_row:
                            raise ValueError("resume record content differs")
                    else:
                        if key in cache:
                            prediction = cache[key]
                            source = "historical_sft" if label == "sft" and suite != "preference_v2_development" else "same_input_cache"
                        else:
                            prediction = generate_prediction(model, tokenizers[label], case, settings["max_new_tokens"])
                            source = "fresh"
                        row = behavior_record(case, prediction, seen, preference=preference,
                            legacy_paired=suite in ("brackets", "spacing_control"), source=source)
                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                        log.flush()
                    cache[key] = row
                    records[label][suite].append(row)
                    print(f"{label} {suite} {index}/{len(cases)} {case.id}: {row['failure'] or 'pass'} ({row['generation_source']})", flush=True)
    summaries, comparisons = {label: {} for label in LABELS}, {}
    contrasts = (("sft", "small_28"), ("sft", "small_168"), ("sft", "large_168"), ("small_28", "small_168"), ("small_168", "large_168"))
    for left, right in contrasts:
        name = f"{left}_vs_{right}"
        comparisons[name] = {}
        for suite in suites:
            result, changes = compare_records(records[left][suite], records[right][suite], paired=suite in ("brackets", "spacing_control"))
            summaries[left][suite], summaries[right][suite] = result["sft"], result["dpo"]
            if suite == "preference_v2_development":
                for label in (left, right):
                    summaries[label][suite]["v2_subgroups"] = v2_subgroups(records[label][suite])
            comparisons[name][suite] = {key: value for key, value in result.items() if key not in ("sft", "dpo")}
            changes = [{("before_"+key[4:] if key.startswith("sft_") else "after_"+key[4:] if key.startswith("dpo_") else key): value for key, value in row.items()} for row in changes]
            write_json(output / f"{name}_{suite}_changes.json", changes)
    counts = {label: dict(Counter(row["generation_source"] for group in records[label].values() for row in group)) for label in LABELS}
    result = {**inventory, "runs": training["runs"], "training_comparison": str(training_path),
        "historical_behavior": str(historical_path), "historical_sft_probes": probes,
        "generation_sources": counts, "models": summaries, "comparisons": comparisons,
        "prediction_files_sha256": {f"{label}_{suite}.jsonl": file_hash(output / f"{label}_{suite}.jsonl") for label in LABELS for suite in suites}}
    write_json(output / "summary.json", result)
    lines = ["# v2 DPO规模对照：实际生成行为", "", "所有组为开发诊断，没有最终未查看测试。SFT旧90条复用已核对记录并用3题重现探针确认，新56条与DPO三分支实际生成；同一模型的相同输入缓存复用。", "",
        "| 数据组 | SFT严格 | A小集28步 | B小集168步 | C大集168步 |", "| --- | --- | --- | --- | --- |"]
    for suite in suites:
        cells = []
        for label in LABELS:
            metric = summaries[label][suite]["metrics"]["end_to_end_pass"]
            cells.append(f"{metric['passed']}/{metric['total']}")
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines += ["", "| 模型 | 新开发算术数值通过 | 新开发算术成对数值通过 | 新开发问候严格 |", "| --- | --- | --- | --- |"]
    for label in LABELS:
        groups = summaries[label]["preference_v2_development"]["v2_subgroups"]
        semantic, paired, greeting = groups["arithmetic"]["semantic_pass"], groups["arithmetic"]["pair_semantic_pass"], groups["greetings"]["metrics"]["end_to_end_pass"]
        lines.append(f"| {label} | {semantic['passed']}/{semantic['total']} | {paired['passed']}/{paired['total']} | {greeting['passed']}/{greeting['total']} |")
    for suite in suites:
        lines += ["", f"## {suite} 逐题输出", "", "| ID | 题目 | 期望表达式 | SFT | A | B | C |", "| --- | --- | --- | --- | --- | --- | --- |"]
        indexed = {label: {row["id"]: row for row in records[label][suite]} for label in LABELS}
        for case in suites[suite]:
            lines.append("| " + " | ".join(table_text(value) for value in (case.id, case.prompt, case.expected_expression,
                *(indexed[label][case.id]["prediction"] for label in LABELS))) + " |")
    lines += ["", "严格AST与当前数字数值通过不同，成对要求互补题同时通过。问候只检验结构与路由。不同数据组含重复问题，不合并成独立测试准确率；单种子与单次子集不证明普遍优劣。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "generation_sources": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
