"""Compare an anchored C rerun with the verified SFT/pure-DPO predictions."""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, table_text, write_json
from freeze_sft_baseline import verify
from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.data.validation import target_for_example
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.scaling import GENERATION_FIELDS, behavior_record, training_keys, v2_subgroups
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import EvalCase

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_control(original, repaired, original_history, repaired_history, expected_anchor_weight=1.0, expected_learning_rate=None):
    """Only the declared objective weight or learning rate may change."""
    ignored = {"experiment_id", "chosen_nll_weight"} if expected_learning_rate is None else {"experiment_id", "learning_rate"}
    configs = [{k: v for k, v in item["config"].items() if k not in ignored} for item in (original, repaired)]
    if configs[0] != configs[1]:
        raise ValueError("repair must match the declared training control setup")
    if expected_learning_rate is None:
        if original["config"].get("chosen_nll_weight", 0) != 0:
            raise ValueError("anchor control must start from pure DPO")
    elif (original["config"].get("chosen_nll_weight") != expected_anchor_weight
          or repaired["config"]["learning_rate"] != expected_learning_rate):
        raise ValueError("learning rate differs from the declared repair")
    if repaired["config"].get("chosen_nll_weight") != expected_anchor_weight:
        raise ValueError("chosen_nll_weight differs from the declared repair")
    for key in ("model", "resolved_revision", "baseline_manifest_sha256", "data_manifest",
                "initial_policy_sha256", "initial_reference_sha256", "before", "training_token_counts"):
        if original[key] != repaired[key]:
            raise ValueError(f"controlled training mismatch: {key}")
    for key in ("reference_unchanged", "reference_no_grad", "policy_frozen_parameters_unchanged",
                "policy_updated", "initial_policy_reference_match", "reload_logits_match"):
        if not repaired[key]:
            raise ValueError(f"repair invariant failed: {key}")
    if len(original_history) != len(repaired_history) or len(repaired_history) != repaired["config"]["max_steps"]:
        raise ValueError("training history length differs")
    for left, right in zip(original_history, repaired_history):
        for key in ("step", "pair_index", "pair_id", "epoch", "chosen_response_tokens",
                    "rejected_response_tokens", "input_tokens", "padded_input_tokens", "learning_rate"):
            if key == "learning_rate" and expected_learning_rate is not None:
                if left[key] != original["config"][key] or right[key] != expected_learning_rate:
                    raise ValueError("training learning rate differs")
                continue
            if left[key] != right[key]:
                raise ValueError(f"training exposure differs: {key}")


@torch.no_grad()
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="同条件核验DPO+chosen NLL修复，不重新训练")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--original-behavior", type=Path, required=True)
    parser.add_argument("--expected-anchor-weight", type=float, default=1.0)
    parser.add_argument("--training-control-run", type=Path)
    parser.add_argument("--expected-learning-rate", type=float)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel

    run, previous = args.run.resolve(), args.original_behavior.resolve()
    repaired, old_behavior = read_json(run / "summary.json"), read_json(previous / "summary.json")
    pure_run = Path(old_behavior["runs"]["large_168"])
    original = read_json(pure_run / "summary.json")
    if file_hash(pure_run / "adapter/adapter_model.safetensors") != old_behavior["adapter_sha256"]["large_168"]:
        raise ValueError("pure DPO adapter differs")
    training_inventory = read_json(Path(old_behavior["training_comparison"]) / "summary.json")
    if (file_hash(Path(old_behavior["training_comparison"]) / "summary.json") != old_behavior["training_summary_sha256"]
        or file_hash(pure_run / "summary.json") != training_inventory["summary_sha256"]["large_168"]):
        raise ValueError("original training identity differs")
    control_run = args.training_control_run.resolve() if args.training_control_run else pure_run
    control = read_json(control_run / "summary.json")
    if bool(args.training_control_run) != (args.expected_learning_rate is not None):
        raise ValueError("learning rate repair requires an explicit training control and rate")
    if args.training_control_run:
        verify_control(original, control, read_rows(pure_run / "train_log.jsonl"), read_rows(control_run / "train_log.jsonl"), args.expected_anchor_weight)
    verify_control(control, repaired, read_rows(control_run / "train_log.jsonl"), read_rows(run / "train_log.jsonl"),
                   args.expected_anchor_weight, args.expected_learning_rate)
    baseline = ROOT / repaired["config"]["baseline"]
    verify(baseline)
    if file_hash(baseline / "manifest.json") != repaired["baseline_manifest_sha256"]:
        raise ValueError("baseline identity differs")
    baseline_manifest = read_json(baseline / "manifest.json")
    for path, digest in old_behavior["source_sha256"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"evaluation source differs: {path}")
    for name, expected in old_behavior["versions"].items():
        if version(name) != expected or repaired["versions"][name] != expected:
            raise ValueError("evaluation runtime differs")
    samples = load_preference_splits(ROOT / repaired["config"]["preference_data"],
        repaired["data_manifest"], repaired["config"]["train_file"])
    prefs = {row.id: row for row in samples["development"]}
    seen = {"sft": training_keys([target_for_example(row) for row in load_examples(
        baseline / "project" / baseline_manifest["training_summary"]["config"]["train_data"])]),
        "repair": training_keys([row.target() for row in samples["train"]])}
    records = {label: {} for label in ("sft", "pure_dpo", "anchored_dpo")}
    suites = {}
    for suite in old_behavior["models"]["sft"]:
        for label, old_label in (("sft", "sft"), ("pure_dpo", "large_168")):
            filename = f"{old_label}_{suite}.jsonl"
            if file_hash(previous / filename) != old_behavior["prediction_files_sha256"][filename]:
                raise ValueError("original prediction hash differs")
            selected = read_rows(previous / filename)
            records[label][suite] = selected
        cases = [EvalCase(**{key: row[key] for key in ("id", "category", "prompt", "expected_type",
            "expected_expression", "expected_result", "system")}) for row in records["sft"][suite]]
        for label in ("sft", "pure_dpo"):
            for row, case in zip(records[label][suite], cases):
                if any(row[key] != value for key, value in asdict(case).items()):
                    raise ValueError("original evaluation cases differ")
                scored = behavior_record(case, row, {})
                for key in ("json_valid", "schema_valid", "route_correct", "result_correct", "end_to_end_pass", "semantic_pass", "failure"):
                    if scored[key] != row[key]:
                        raise ValueError("original scores differ")
        if suite == "preference_v2_development" and cases != [row.target() for row in samples["development"]]:
            raise ValueError("repair development set differs")
        suites[suite] = cases
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(run / "tokenizer"), local_files_only=True, trust_remote_code=False))
    old_tokenizer = AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"), local_files_only=True, trust_remote_code=False)
    if (tokenizer.raw.get_vocab() != old_tokenizer.get_vocab() or tokenizer.raw.chat_template != old_tokenizer.chat_template
        or tokenizer.raw.special_tokens_map != old_tokenizer.special_tokens_map
        or tokenizer.raw.pad_token_id != old_tokenizer.pad_token_id or tokenizer.raw.eos_token_id != old_tokenizer.eos_token_id):
        raise ValueError("tokenizers differ")
    settings = old_behavior["generation"]
    if settings != baseline_manifest["generation"]:
        raise ValueError("generation settings differ")
    set_seed(settings["seed"])
    torch.set_num_threads(settings["cpu_threads"])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/evaluations" / f"dpo-anchor-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    inventory = {"run": str(run), "original_behavior": str(previous), "original_summary_sha256": file_hash(previous / "summary.json"),
        "training_summary_sha256": file_hash(run / "summary.json"), "adapter_sha256": file_hash(run / "adapter/adapter_model.safetensors"),
        "baseline_manifest_sha256": repaired["baseline_manifest_sha256"], "generation": settings,
        "chosen_nll_weight": args.expected_anchor_weight,
        "training_control_run": str(control_run), "training_control_summary_sha256": file_hash(control_run / "summary.json"),
        "declared_learning_rate": args.expected_learning_rate,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in (Path(__file__),
            ROOT / "src/agentic_post_training/evaluation/scaling.py", ROOT / "src/agentic_post_training/evaluation/generation.py")}}
    write_json(output / "run_manifest.json", inventory)
    print(json.dumps({"output": str(output), "controls_verified": True}, ensure_ascii=False), flush=True)
    base = AutoModelForCausalLM.from_pretrained(repaired["model"], revision=repaired["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(run / "adapter"), is_trainable=False)
    model.requires_grad_(False)
    model.eval()
    cache = {}
    for suite, cases in suites.items():
        rows = []
        with (output / f"anchored_dpo_{suite}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for index, case in enumerate(cases, 1):
                messages = [{"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}]
                if tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True) != old_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True):
                    raise ValueError("tokenized prompts differ")
                key = (case.system, case.prompt)
                cached = key in cache
                generation = cache[key] if cached else generate_prediction(model, tokenizer, case, settings["max_new_tokens"])
                cache[key] = generation
                row = behavior_record(case, generation, seen, preference=prefs.get(case.id) if suite == "preference_v2_development" else None,
                    legacy_paired=suite in ("brackets", "spacing_control"), source="same_input_cache" if cached else "fresh")
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{suite} {index}/{len(cases)}: {row['failure'] or 'pass'}", flush=True)
        records["anchored_dpo"][suite] = rows
    summaries, comparisons = {label: {} for label in records}, {}
    for before in ("sft", "pure_dpo"):
        comparisons[before] = {}
        for suite in suites:
            result, changes = compare_records(records[before][suite], records["anchored_dpo"][suite], paired=suite in ("brackets", "spacing_control"))
            for label, metric_label in ((before, "sft"), ("anchored_dpo", "dpo")):
                summaries[label][suite] = result[metric_label]
                if suite == "preference_v2_development":
                    summaries[label][suite]["v2_subgroups"] = v2_subgroups(records[label][suite])
            comparisons[before][suite] = {key: value for key, value in result.items() if key not in ("sft", "dpo")}
            changes = [{("before_"+key[4:] if key.startswith("sft_") else "after_"+key[4:] if key.startswith("dpo_") else key): value for key, value in row.items()} for row in changes]
            write_json(output / f"{before}_vs_anchored_{suite}_changes.json", changes)
    for label in ("sft", "pure_dpo"):
        for suite, rows in records[label].items():
            (output / f"{label}_{suite}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8", newline="\n")
    metrics = summaries["anchored_dpo"]["preference_v2_development"]
    passes = metrics["metrics"]["end_to_end_pass"]["passed"]
    semantic = metrics["semantic_pass"]["passed"]
    gates = {"new_strict_at_least_sft": passes >= 36, "new_semantic_at_least_sft": semantic >= 44,
        "new_json_schema_route_all_valid": all(metrics["metrics"][key]["passed"] == 56 for key in ("json_valid", "schema_valid", "route_correct")),
        "legacy_strict_not_lower_than_sft": all(summaries["anchored_dpo"][suite]["metrics"]["end_to_end_pass"]["passed"] >= bound for suite, bound in (("tool_calls", 19), ("brackets", 33), ("spacing_control", 12), ("preference_v1_development", 7))),
        "new_strict_or_semantic_exceeds_sft": passes > 36 or semantic > 44}
    result = {**inventory, "controls_verified": True, "models": summaries, "comparisons": comparisons,
        "generation_sources": dict(Counter(row["generation_source"] for group in records["anchored_dpo"].values() for row in group)),
        "acceptance_checks": gates, "all_acceptance_checks_pass": all(gates.values()),
        "prediction_files_sha256": {f"{label}_{suite}.jsonl": file_hash(output / f"{label}_{suite}.jsonl") for label in records for suite in suites}}
    write_json(output / "summary.json", result)
    description = (f"与原C相比仅增加chosen token平均NLL权重{args.expected_anchor_weight:g}。" if args.expected_learning_rate is None else
        f"与权重{args.expected_anchor_weight:g}上一轮相比仅将学习率降低至{args.expected_learning_rate:g}；与原C相比同时增加监督项与降低学习率。")
    lines = ["# DPO正确答案监督修复：受控生成对照", "", description + "同一数据、起点、顺序、步数。全部为已查看开发诊断。", "",
        "| 数据组 | SFT严格 | 原C纯DPO | DPO+chosen NLL |", "| --- | --- | --- | --- |"]
    for suite in suites:
        cells = [f"{summaries[label][suite]['metrics']['end_to_end_pass']['passed']}/{len(suites[suite])}" for label in records]
        lines.append(f"| {suite} | " + " | ".join(cells) + " |")
    lines += ["", "验收检查：", "", "```json", json.dumps(gates, ensure_ascii=False, indent=2), "```"]
    for suite, cases in suites.items():
        lines += ["", f"## {suite}", "", "| ID | 输入 | SFT | 原C | 修复模型 |", "| --- | --- | --- | --- | --- |"]
        for index, case in enumerate(cases):
            lines.append("| " + " | ".join(table_text(value) for value in (case.id, case.prompt, *(records[label][suite][index]["prediction"] for label in records))) + " |")
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "acceptance_checks": gates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
