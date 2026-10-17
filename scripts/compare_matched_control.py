"""Common generation and model-free re-audit for the frozen three-arm plan."""

import argparse
import gc
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, write_json
from freeze_sft_baseline import verify
from matched_control_common import ARMS, DATA, PLAN_HASH, ROOT, admission_checks, load_design, read_json, read_rows, verify_exposures
from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.validation import target_for_example
from agentic_post_training.evaluation.comparison import compare_records
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.scaling import GENERATION_FIELDS, behavior_record, training_keys, v2_subgroups
from agentic_post_training.evaluation.tool_calls import EvalCase, score_output

LABELS = ("baseline", *ARMS)


def verify_sources(inventory):
    for name, expected in inventory["source_sha256"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"source changed: {name}")


def case_from_row(row):
    return EvalCase(**{field.name: row[field.name] for field in fields(EvalCase)})


def build_results(records, plan):
    summaries = {label: {} for label in LABELS}
    comparisons, changed_files = {}, {}
    for left, right in combinations(LABELS, 2):
        contrast = f"{left}_vs_{right}"
        comparisons[contrast] = {}
        for suite in plan["evaluation"]["suites"]:
            result, changes = compare_records(records[left][suite], records[right][suite], paired=suite in ("brackets", "spacing_control"))
            for label, key in ((left, "sft"), (right, "dpo")):
                summaries[label][suite] = result[key]
                if suite == "preference_v2_development":
                    summaries[label][suite]["v2_subgroups"] = v2_subgroups(records[label][suite])
            comparisons[contrast][suite] = {key: value for key, value in result.items() if key not in ("sft", "dpo")}
            changes = [{("before_"+key[4:] if key.startswith("sft_") else "after_"+key[4:] if key.startswith("dpo_") else key): value for key, value in row.items()} for row in changes]
            changed_files[f"{contrast}_{suite}_changes.json"] = changes
    checks = {label: admission_checks(records["baseline"], records[label], plan,
        None if label == "chosen_sft" else records["chosen_sft"]) for label in ARMS}
    return {"models": summaries, "comparisons": comparisons, "admission": checks,
        "any_dpo_added_value": any(checks[label]["conditional_dpo_added_value_pass"] for label in ARMS[1:])}, changed_files


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="统一评估冻结38同题三组；可不加载模型重审结果")
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--require-admission", action="store_true")
    args = parser.parse_args()
    if (args.resume or args.audit_only) and args.output is None:
        parser.error("resume/audit-only requires output")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    plan, _, baseline, samples = load_design()
    training_run = args.training_run.resolve()
    training = read_json(training_run / "summary.json")
    if training["plan_sha256"] != PLAN_HASH or training["plan"] != plan or training["data_manifest_sha256"] != file_hash(DATA / "manifest.json"):
        raise ValueError("training plan identity differs")
    verify_sources(training)
    if not training["matching_exposure_verified"]:
        raise ValueError("exposure control missing")
    histories = {}
    for label in ARMS:
        path = training_run / label
        saved = read_json(path / "summary.json")
        if (file_hash(path / "summary.json") != training["arm_summary_sha256"][label]
            or saved != training["arms"][label] or saved["plan_sha256"] != PLAN_HASH
            or file_hash(path / "adapter/adapter_model.safetensors") != saved["adapter_sha256"]
            or file_hash(path / "train_log.jsonl") != saved["train_log_sha256"]):
            raise ValueError("arm artifacts differ")
        if any(not saved[key] for key in ("policy_frozen_parameters_unchanged", "policy_updated", "reference_unchanged", "reference_no_grad", "reload_logits_match")):
            raise ValueError("training invariant failed")
        histories[label] = read_rows(path / "train_log.jsonl")
    if len({training["arms"][label]["initial_policy_sha256"] for label in ARMS}) != 1 or any(training["arms"][label]["initial_policy_sha256"] != training["arms"][label]["initial_reference_sha256"] for label in ARMS):
        raise ValueError("initial weights differ")
    verify_exposures(histories, samples["chosen_sft"], plan)
    cache_path = training_run / "reference_log_probs.pt"
    if file_hash(cache_path) != training["reference_cache_sha256"]:
        raise ValueError("reference cache hash differs")
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    if set(cached) != set(ARMS[1:]) or any(len(values) != 38 or any(value.shape != (2,) or value.requires_grad or not torch.isfinite(value).all().item() for value in values) for values in cached.values()):
        raise ValueError("invalid reference cache")
    historical = ROOT / plan["evaluation"]["historical_behavior"]
    if file_hash(historical / "summary.json") != plan["evaluation"]["historical_summary_sha256"]:
        raise ValueError("historical summary differs")
    old_summary = read_json(historical / "summary.json")
    verify_sources(old_summary)
    for name, expected in old_summary["versions"].items():
        if version(name) != expected or training["versions"][name] != expected:
            raise ValueError("runtime versions differ")
    prefs = {row.id: row for row in samples["development"]}
    baseline_manifest = read_json(baseline / "manifest.json")
    sft_cases = [target_for_example(row) for row in load_examples(baseline / "project" / baseline_manifest["training_summary"]["config"]["train_data"])]
    seen = {"sft": training_keys(sft_cases), "matched": training_keys([row.target() for row in samples["real_dpo"]])}
    suites, records = {}, {label: {} for label in LABELS}

    def record(case, generation, suite, source):
        return behavior_record(case, generation, seen, preference=prefs.get(case.id) if suite == "preference_v2_development" else None,
            legacy_paired=suite in ("brackets", "spacing_control"), source=source)

    for suite, count in plan["evaluation"]["suites"].items():
        filename = f"sft_{suite}.jsonl"
        if file_hash(historical / filename) != plan["evaluation"]["historical_sft_prediction_sha256"][filename]:
            raise ValueError("historical predictions differ")
        old = read_rows(historical / filename)
        if len(old) != count or len({row["id"] for row in old}) != count:
            raise ValueError("historical case count differs")
        suites[suite] = [case_from_row(row) for row in old]
        rows = []
        for case, saved in zip(suites[suite], old):
            actual = record(case, saved, suite, "historical_sft")
            if any(actual[key] != saved[key] for key in (*score_output(saved["prediction"], case), "semantic_pass")):
                raise ValueError("historical score differs")
            rows.append(actual)
        records["baseline"][suite] = rows
    if suites["preference_v2_development"] != [row.target() for row in samples["development"]]:
        raise ValueError("development targets differ")
    baseline_check = admission_checks(records["baseline"], records["baseline"], plan)
    if not baseline_check["recovery_pass"]:
        raise ValueError("historical baseline counts do not match plan")
    sources = [Path(__file__), ROOT / "scripts/matched_control_common.py", *[ROOT / "src/agentic_post_training/evaluation" / name
        for name in ("generation.py", "scaling.py", "tool_calls.py", "comparison.py", "brackets.py")]]
    inventory = {"plan_sha256": PLAN_HASH, "training_run": str(training_run),
        "training_summary_sha256": file_hash(training_run / "summary.json"),
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
        "generation": plan["evaluation"]["generation"], "versions": old_summary["versions"]}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else ROOT / "artifacts/evaluations" / f"matched-control-{stamp}-{uuid4().hex[:8]}"
    if args.resume or args.audit_only:
        if read_json(output / "run_manifest.json") != inventory:
            raise ValueError("evaluation inventory changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", inventory)
    print(json.dumps({"output": str(output), "mode": "audit" if args.audit_only else "generation"}, ensure_ascii=False), flush=True)
    probe_suites = ("tool_calls", "brackets", "spacing_control")
    probes = []
    if not args.audit_only:
        verify(baseline)
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import PeftModel
        settings = inventory["generation"]
        torch.set_num_threads(settings["cpu_threads"])
        set_seed(settings["seed"])
        tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"), local_files_only=True, trust_remote_code=False))
        load_design(tokenizer)

        def load_model(adapter):
            base = AutoModelForCausalLM.from_pretrained(training["model"], revision=training["resolved_revision"],
                cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation="eager").cpu()
            model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
            model.requires_grad_(False)
            model.eval()
            return model

        if args.resume and (output / "baseline_probes.json").exists():
            probes = read_json(output / "baseline_probes.json")
        else:
            model = load_model(baseline / "checkpoint/adapter")
            for suite in probe_suites:
                case = suites[suite][0]
                generation = generate_prediction(model, tokenizer, case, settings["max_new_tokens"])
                if any(generation[key] != records["baseline"][suite][0][key] for key in GENERATION_FIELDS):
                    raise ValueError("historical baseline fresh probe differs")
                probes.append({"suite": suite, "id": case.id, **generation})
            write_json(output / "baseline_probes.json", probes)
            del model
            gc.collect()
        for label in ARMS:
            raw = AutoTokenizer.from_pretrained(str(training_run / label / "tokenizer"), local_files_only=True, trust_remote_code=False)
            if (raw.get_vocab() != tokenizer.raw.get_vocab() or raw.chat_template != tokenizer.raw.chat_template
                or raw.special_tokens_map != tokenizer.raw.special_tokens_map or raw.pad_token_id != tokenizer.pad_token_id
                or raw.eos_token_id != tokenizer.raw.eos_token_id):
                raise ValueError("arm tokenizer differs")
            model = load_model(training_run / label / "adapter")
            cache = {}
            for suite, cases in suites.items():
                path = output / f"{label}_{suite}.jsonl"
                done = read_rows(path) if args.resume and path.exists() else []
                if len(done) > len(cases) or [row["id"] for row in done] != [case.id for case in cases[:len(done)]]:
                    raise ValueError("resume records must be an ordered prefix")
                rows = []
                with path.open("a" if done else "w", encoding="utf-8", newline="\n") as stream:
                    for index, case in enumerate(cases):
                        key = (case.system, case.prompt)
                        if index < len(done):
                            row = record(case, done[index], suite, done[index]["generation_source"])
                            if row != done[index]:
                                raise ValueError("resume record differs")
                        else:
                            reused = key in cache
                            generation = cache[key] if reused else generate_prediction(model, tokenizer, case, settings["max_new_tokens"])
                            row = record(case, generation, suite, "same_input_cache" if reused else "fresh")
                            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                            stream.flush()
                            print(f"{label} {suite} {index+1}/{len(cases)}: {row['failure'] or 'pass'}", flush=True)
                        if key in cache and any(cache[key][field] != row[field] for field in GENERATION_FIELDS):
                            raise ValueError("same input has inconsistent generation")
                        cache[key] = row
                        rows.append(row)
                records[label][suite] = rows
            del model
            gc.collect()
        for suite, rows in records["baseline"].items():
            (output / f"baseline_{suite}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8", newline="\n")
    else:
        probes = read_json(output / "baseline_probes.json")
        for label in LABELS:
            for suite, cases in suites.items():
                rows = read_rows(output / f"{label}_{suite}.jsonl")
                if len(rows) != len(cases):
                    raise ValueError("incomplete evaluation")
                for row, case in zip(rows, cases):
                    if any(row[key] != value for key, value in asdict(case).items()) or record(case, row, suite, row["generation_source"]) != row:
                        raise ValueError("saved prediction/score/metadata differs")
                if label == "baseline" and rows != records[label][suite]:
                    raise ValueError("saved historical baseline differs")
                records[label][suite] = rows
    if [row["suite"] for row in probes] != list(probe_suites) or any(row["id"] != suites[row["suite"]][0].id or any(row[key] != records["baseline"][row["suite"]][0][key] for key in GENERATION_FIELDS) for row in probes):
        raise ValueError("baseline probes differ")
    calculated, changes = build_results(records, plan)
    summary = {**inventory, **calculated, "prediction_files_sha256": {
        f"{label}_{suite}.jsonl": file_hash(output / f"{label}_{suite}.jsonl") for label in LABELS for suite in suites},
        "generation_sources": {label: dict(Counter(row["generation_source"] for rows in records[label].values() for row in rows)) for label in LABELS},
        "baseline_probes_sha256": file_hash(output / "baseline_probes.json"), "baseline_replaced": False}
    if args.audit_only:
        if summary != read_json(output / "summary.json") or any(value != read_json(output / name) for name, value in changes.items()):
            raise ValueError("saved comparisons, gates or hashes differ")
        write_json(output / "verification.json", {"artifact_audit_passed": True, "records_rescored": 584,
            "contrasts_recomputed": 6, "changes_files_recomputed": len(changes), "plan_sha256": PLAN_HASH,
            "summary_sha256": file_hash(output / "summary.json"), "model_weights_loaded": False,
            "any_dpo_added_value": calculated["any_dpo_added_value"]})
    else:
        for name, value in changes.items():
            write_json(output / name, value)
        write_json(output / "summary.json", summary)
        lines = ["# 38同题三组对照：实际生成结果", "", "固定方案、单种子、单遍38步；全部为已查看开发诊断。未替换基线。", "",
            "| 数据组 | 冻结SFT | 继续SFT | 合成负例DPO | 真实负例DPO |", "| --- | --- | --- | --- | --- |"]
        for suite in suites:
            cells = [f"{calculated['models'][label][suite]['metrics']['end_to_end_pass']['passed']}/{len(suites[suite])}" for label in LABELS]
            lines.append(f"| {suite} | " + " | ".join(cells) + " |")
        lines += ["", "## 预先质量条件", "", "```json", json.dumps(calculated["admission"], ensure_ascii=False, indent=2), "```",
            "", "训练预算按chosen曝光对齐，不是token算力相等；26个负例改变、12个不变。逐题变化见30份changes，不能用固定偏好loss代替生成结论。"]
        (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "audit_only": args.audit_only,
        "any_dpo_added_value": calculated["any_dpo_added_value"], "admission": calculated["admission"]}, ensure_ascii=False), flush=True)
    if args.require_admission and not calculated["any_dpo_added_value"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
