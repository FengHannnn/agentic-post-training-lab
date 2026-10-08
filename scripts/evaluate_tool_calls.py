"""Compare a saved adapter and its exact base on held-out tool requests."""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from agentic_post_training.data import SFTExample, load_examples
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.evaluation.generation import generate_prediction
from agentic_post_training.evaluation.tool_calls import load_eval_cases, score_output, summarize
from agentic_post_training.trainers.sft_lora import check_split


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="比较基础模型与 LoRA 的独立工具调用评测")
    parser.add_argument("--run", type=Path, required=True, help="含 summary.json、adapter 和 tokenizer 的训练结果目录")
    parser.add_argument("--data", type=Path, default=root / "data/eval/tool_calls_v1.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    run = args.run.resolve()
    training_summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    config = training_summary["config"]
    cases = load_eval_cases(args.data)
    heldout = [SFTExample(case.id, case.system, case.prompt, "evaluation only") for case in cases]
    check_split(load_examples(root / config["train_data"]), heldout)
    check_split(load_examples(root / config["eval_data"]), heldout)
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(run / "tokenizer"), trust_remote_code=False))
    base = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=training_summary["resolved_revision"],
        cache_dir=str(root / ".cache/huggingface"), local_files_only=args.local_files_only,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager",
    ).cpu()
    name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = root / "artifacts/evaluations" / ("tool-calls-" + name)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for label in ("base", "lora"):
        model = base if label == "base" else PeftModel.from_pretrained(base, str(run / "adapter"), is_trainable=False)
        records = []
        with (output / f"{label}_predictions.jsonl").open("w", encoding="utf-8") as destination:
            for index, case in enumerate(cases, 1):
                generation = generate_prediction(model, tokenizer, case, args.max_new_tokens)
                record = {
                    "id": case.id, "category": case.category, "prompt": case.prompt,
                    "expected_type": case.expected_type, "expected_expression": case.expected_expression,
                    "expected_result": case.expected_result, **generation,
                    **score_output(generation["prediction"], case),
                }
                records.append(record)
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
                print(f"{label} {index}/{len(cases)} {case.id}: {record['failure'] or 'pass'}", flush=True)
        summaries[label] = summarize(records)
        summaries[label]["hit_token_limit_count"] = sum(row["hit_token_limit"] for row in records)
        summaries[label]["by_category"] = {
            category: summarize([row for row in records if row["category"] == category])
            for category in sorted({row["category"] for row in records})
        }
    result = {
        "training_run": str(run), "model": config["model"],
        "resolved_revision": training_summary["resolved_revision"], "training_steps": config["max_steps"],
        "evaluation_data": str(args.data.resolve()),
        "evaluation_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "seed": config["seed"],
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        **summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 工具调用独立评测", "", f"训练结果：`{run.name}`；训练步数：{config['max_steps']}。", "", "| 指标 | 基础模型 | LoRA |", "| --- | --- | --- |"]
    names = {"json_valid": "JSON 合法", "schema_valid": "结构合法", "route_correct": "路由正确", "arguments_valid": "参数可执行（计算题）", "arguments_correct": "参数结构正确（计算题）", "tool_execution_success": "工具执行成功（计算题）", "result_correct": "计算结果正确（计算题）", "end_to_end_pass": "请求执行闭环通过"}
    for metric, title in names.items():
        cells = []
        for label in ("base", "lora"):
            value = summaries[label]["metrics"][metric]
            cells.append(f"{value['passed']}/{value['total']} ({value['rate']:.1%})" if value["total"] else "不适用")
        lines.append(f"| {title} | {cells[0]} | {cells[1]} |")
    lines += ["", "参数正确采用算术 AST 结构比较，忽略空白与多余括号，但不接受所有数学等价改写。问候题只检查 final 路由、结构和非空回答，不评判回答语义。", "", "模型输出只去掉 tokenizer 特殊 token，不修复 JSON。闭环指请求解析和本地计算器执行，未包含第二轮模型最终答复。", "", "20 条题目只能提供初步诊断；本次查看后将它们用于调参时，它们就成为开发集，需要另建最终测试集。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("结果目录：", output, flush=True)


if __name__ == "__main__":
    main()
