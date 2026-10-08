"""Compare two saved LoRA adapters on bracket and wording contrast pairs."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from agentic_post_training.data import load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.environments.calculator import expression_structure
from agentic_post_training.evaluation.brackets import diagnostic_summary
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import load_eval_cases, score_output


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="括号与不同文字问法的成对诊断")
    parser.add_argument("--run-v1", type=Path, required=True)
    parser.add_argument("--run-v2", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=root / "data/eval/bracket_diagnostic_v1.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    runs = {"v1": args.run_v1.resolve(), "v2": args.run_v2.resolve()}
    saved = {label: json.loads((path / "summary.json").read_text(encoding="utf-8")) for label, path in runs.items()}
    if len({(data["config"]["model"], data["resolved_revision"]) for data in saved.values()}) != 1:
        raise ValueError("diagnostic adapters must share the exact same base")
    tokenizers = {label: HFChatTokenizer(AutoTokenizer.from_pretrained(str(path / "tokenizer"), trust_remote_code=False)) for label, path in runs.items()}
    if tokenizers["v1"].raw.get_vocab() != tokenizers["v2"].raw.get_vocab() or tokenizers["v1"].raw.chat_template != tokenizers["v2"].raw.chat_template:
        raise ValueError("diagnostic runs must share tokenizer and template")
    config = saved["v1"]["config"]
    set_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    cases = load_eval_cases(args.data)
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=saved["v1"]["resolved_revision"],
        cache_dir=str(root / ".cache/huggingface"), local_files_only=args.local_files_only,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager",
    ).cpu()
    model = PeftModel.from_pretrained(base, str(runs["v1"] / "adapter"), adapter_name="v1", is_trainable=False)
    model.load_adapter(str(runs["v2"] / "adapter"), adapter_name="v2", is_trainable=False)
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = root / "artifacts/evaluations" / ("bracket-diagnostic-" + name)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    all_records = {}
    for label in ("v1", "v2"):
        model.set_adapter(label)
        model.eval()
        train = load_examples(root / saved[label]["config"]["train_data"])
        seen_prompts = {(item.system, item.prompt) for item in train}
        seen_expressions = {
            expression_structure(payload["arguments"]["expression"])
            for item in train for payload in [json.loads(item.response)] if payload["type"] == "tool_call"
        }
        records = []
        with (output / f"{label}_predictions.jsonl").open("w", encoding="utf-8") as destination:
            for index, case in enumerate(cases, 1):
                number_set, wording, form = case.category.split("/")
                generation = generate_prediction(model, tokenizers[label], case, args.max_new_tokens)
                score = score_output(generation["prediction"], case)
                record = {
                    "id": case.id, "category": case.category, "prompt": case.prompt,
                    "expected_type": case.expected_type, "expected_expression": case.expected_expression,
                    "expected_result": case.expected_result, "number_set": number_set,
                    "wording": wording, "form": form,
                    "pair_id": f"{number_set}/{wording}/{form.split('_')[0]}",
                    "seen_training_prompt": (case.system, case.prompt) in seen_prompts,
                    "seen_training_expression": expression_structure(case.expected_expression) in seen_expressions,
                    **generation, **score,
                    "semantic_pass": score["schema_valid"] and score["route_correct"] and score["result_correct"],
                }
                records.append(record)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
                print(f"{label} {index}/{len(cases)} {case.category}: {score['failure'] or 'pass'}", flush=True)
        all_records[label] = records
        summaries[label] = diagnostic_summary(records)
    result = {
        "runs": {label: str(path) for label, path in runs.items()},
        "training_steps": {label: data["steps_completed"] for label, data in saved.items()},
        "resolved_revision": saved["v1"]["resolved_revision"],
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "seed": config["seed"],
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        **summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 括号与文字问法诊断", "", "同时报告严格 AST 与数值语义；数值正确不证明所有输入上的代数等价。已见训练样本探针不是泛化测试。", "", "| 指标 | v1 | v2 |", "| --- | --- | --- |"]
    for title, key in (("严格通过", "strict"), ("数值语义正确", "semantic"), ("成对数值均正确", "pair")):
        cells = []
        for label in ("v1", "v2"):
            value = summaries[label]["metrics"]["end_to_end_pass"] if key == "strict" else summaries[label]["semantic_pass"] if key == "semantic" else summaries[label]["pair_semantic_pass"]
            cells.append(f"{value['passed']}/{value['total']}")
        lines.append(f"| {title} | {cells[0]} | {cells[1]} |")
    for field, title in (("wording", "问法"), ("number_set", "数字组"), ("form", "结构")):
        lines += ["", f"## 按{title}比较", "", "| 分组 | v1 数值正确 | v2 数值正确 |", "| --- | --- | --- |"]
        for key in summaries["v1"]["by_" + field]:
            cells = []
            for label in ("v1", "v2"):
                value = summaries[label]["by_" + field][key]
                cells.append(f"{value['semantic_passed']}/{value['total']}")
            lines.append(f"| {key} | {cells[0]} | {cells[1]} |")
    lines += ["", "## 逐题表达式", "", "| 题目 | 期望 | v1 输出 | v2 输出 |", "| --- | --- | --- | --- |"]
    for left, right in zip(all_records["v1"], all_records["v2"]):
        expressions = []
        for row in (left, right):
            try:
                expression = json.loads(row["prediction"])["arguments"]["expression"]
            except (ValueError, KeyError, TypeError):
                expression = row["failure"]
            expressions.append(str(expression).replace("|", "\\|").replace("\n", " "))
        lines.append(f"| {left['category']} | `{left['expected_expression']}` | `{expressions[0]}` | `{expressions[1]}` |")
    lines += ["", "观察按问法、数字和结构的差异，只能提出解释假设，不能由一次小样本诊断确定训练量或模型容量是根因。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("诊断结果：", output, flush=True)


if __name__ == "__main__":
    main()
