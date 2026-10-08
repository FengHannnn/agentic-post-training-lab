"""Collect v2 outputs on new inputs and build audited, sourced preference pairs."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from freeze_sft_baseline import verify
from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import PreferenceExample, semantic_pass, validate_preferences
from agentic_post_training.data.validation import target_for_example
from agentic_post_training.environments.calculator import calculate
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import EvalCase, SYSTEM_PROMPT, load_eval_cases, score_output

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baselines/sft-v2-phase3"


def candidates():
    triples = {"train": [(4, 13, 9), (5, 17, 8), (6, 19, 7), (7, 23, 11), (8, 29, 10), (9, 31, 14)],
               "development": [(10, 37, 16), (11, 43, 18)]}
    for split, numbers in triples.items():
        for a, b, c in numbers:
            family = f"arithmetic-{a}-{b}-{c}"
            forms = [
                ("right_grouped", f"Calculate {a} * ({b} + {c}).", f"{a}*({b}+{c})", f"{a}*{b}+{c}"),
                ("right_plain", f"Multiply {a} by {b} first, then add {c}.", f"{a}*{b}+{c}", f"{a}*({b}+{c})"),
                ("left_grouped", f"Multiply the sum of {b} and {c} by {a}.", f"({b}+{c})*{a}", f"{b}+{c}*{a}"),
                ("left_plain", f"Add {b} to the product of {c} and {a}.", f"{b}+{c}*{a}", f"({b}+{c})*{a}"),
            ]
            for form, prompt, expression, wrong in forms:
                yield split, family, EvalCase(f"dpo-v1-{split}-{a}-{b}-{c}-{form}", form,
                      prompt, "tool_call", expression, calculate(expression)), wrong
    greetings = {"train": ["Hello, happy to see you here!", "Good morning, wishing you a pleasant day!",
                            "Hi there, thanks for your company!", "Good evening, pleased to meet you!"],
                 "development": ["Hello there, hope your day is going well!", "Hi, nice meeting you today!"]}
    for split, prompts in greetings.items():
        for index, prompt in enumerate(prompts, 1):
            family = f"greeting-{split}-{index}"
            yield split, family, EvalCase(f"dpo-v1-{family}", "greeting", prompt, "final"), "1+1"


def tool_text(expression):
    return json.dumps({"type": "tool_call", "name": "calculator",
                       "arguments": {"expression": expression}}, separators=(",", ":"))


def protected_cases():
    protected = []
    for path in sorted((ROOT / "data/eval").glob("*.jsonl")):
        protected += load_eval_cases(path)
    for path in sorted((ROOT / "data/training").glob("sft_*.jsonl")):
        protected += [target_for_example(row) for row in load_examples(path)]
    for path in sorted((ROOT / "data/examples").glob("*_en.jsonl")):
        protected += [target_for_example(row) for row in load_examples(path)]
    return protected


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="构造并审计小规模 DPO 偏好数据")
    parser.add_argument("--output", type=Path, default=ROOT / "data/preferences/dpo_v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    from agentic_post_training.data.preferences import load_preferences
    verify(BASELINE)
    manifest = json.loads((BASELINE / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(
        str(BASELINE / "checkpoint/tokenizer"), local_files_only=True, trust_remote_code=False))
    protected = protected_cases()
    output = args.output.resolve()
    if args.audit_only:
        saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        baseline_hash = hashlib.sha256((BASELINE / "manifest.json").read_bytes()).hexdigest()
        if saved_manifest["baseline_manifest_sha256"] != baseline_hash:
            raise ValueError("Baseline manifest changed since data construction")
        for name, digest in saved_manifest["files"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Preference data changed: {name}")
        rows = {split: load_preferences(output / f"{split}.jsonl") for split in ("train", "development")}
        for samples in rows.values():
            for row in samples:
                provenance = row.provenance
                if provenance["baseline_manifest_sha256"] != baseline_hash:
                    raise ValueError(f"{row.id}: inconsistent baseline provenance")
                if provenance["rejected_source"] == "v2_greedy_generation":
                    if row.rejected != provenance["generation"]["prediction"]:
                        raise ValueError(f"{row.id}: rejected differs from saved model output")
                elif provenance["rejected_source"] != "synthetic_counterexample":
                    raise ValueError(f"{row.id}: unknown rejected source")
        print(json.dumps(validate_preferences(rows["train"], rows["development"], protected,
                                              tokenizer, 192), ensure_ascii=False, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    raw_dir = ROOT / "artifacts/evaluations" / (
        "dpo-preferences-v1" if output == ROOT / "data/preferences/dpo_v1"
        else "dpo-preferences-" + output.name)
    if raw_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing generation log: {raw_dir}")
    items = list(candidates())
    # Check prompt/family/target isolation before spending time on generation.
    provisional = {split: [] for split in ("train", "development")}
    for split, family, case, wrong in items:
        chosen = tool_text(case.expected_expression) if case.expected_type == "tool_call" else '{"type":"final","answer":"Hello!"}'
        provisional[split].append(PreferenceExample(
            case.id, family, case.system, case.prompt, chosen, tool_text(wrong),
            case.category, case.expected_type, case.expected_expression, case.expected_result))
    validate_preferences(provisional["train"], provisional["development"], protected, tokenizer, 192)
    torch.set_num_threads(2)
    set_seed(42)
    base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
        cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager").cpu()
    model = PeftModel.from_pretrained(base, str(BASELINE / "checkpoint/adapter"), is_trainable=False)
    rows = {split: [] for split in provisional}
    raw_records = []
    baseline_hash = hashlib.sha256((BASELINE / "manifest.json").read_bytes()).hexdigest()
    for index, (split, family, case, wrong) in enumerate(items, 1):
        generation = generate_prediction(model, tokenizer, case, 64)
        score = score_output(generation["prediction"], case)
        use_model = bool(generation["prediction"].strip()) and not semantic_pass(score, case.expected_type)
        chosen = tool_text(case.expected_expression) if case.expected_type == "tool_call" else '{"type":"final","answer":"Hello!"}'
        rejected = generation["prediction"] if use_model else tool_text(wrong)
        provenance = {
            "baseline_id": manifest["baseline_id"], "baseline_manifest_sha256": baseline_hash,
            "chosen_source": "human_reviewed_deterministic_template",
            "rejected_source": "v2_greedy_generation" if use_model else "synthetic_counterexample",
            "synthetic_rule": None if use_model else "wrong_grouping" if case.expected_type == "tool_call" else "wrong_route",
            "rejected_failure": score_output(rejected, case)["failure"],
            "generation": generation, "generation_score": score,
        }
        rows[split].append(PreferenceExample(case.id, family, case.system, case.prompt, chosen, rejected,
                          case.category, case.expected_type, case.expected_expression, case.expected_result, provenance))
        raw_records.append({"split": split, "family_id": family, **asdict(case), **generation, **score})
        print(f"{index}/{len(items)} {case.id}: {provenance['rejected_source']}", flush=True)
    audit = validate_preferences(rows["train"], rows["development"], protected, tokenizer, 192)
    output.mkdir(parents=True, exist_ok=False)
    for split, samples in rows.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in samples), encoding="utf-8", newline="\n")
    result = {
        "baseline_id": manifest["baseline_id"], "baseline_manifest_sha256": baseline_hash,
        "model_revision": manifest["resolved_revision"], "max_length": 192,
        "generation": manifest["generation"], "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "audit": audit, "protected_case_records": len(protected),
        "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in ("train.jsonl", "development.jsonl")},
        "limitations": ["开发数据，共享训练措辞模板，不证明模板外泛化", "chosen 标签由确定性模板产生并人工复查", "rejected 混合真实模型错误与合成对照", "没有最终未查看测试集"],
    }
    (output / "manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    raw_dir.mkdir(parents=True, exist_ok=False)
    (raw_dir / "v2_predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_records), encoding="utf-8", newline="\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
