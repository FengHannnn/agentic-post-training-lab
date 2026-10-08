"""Audit expanded labels, split overlap and actual token lengths."""

import argparse
import json
import sys
from pathlib import Path

from agentic_post_training.data import encode_sft, load_examples
from agentic_post_training.data.hf_tokenizer import load_hf_tokenizer
from agentic_post_training.data.validation import target_for_example, validate_dataset
from agentic_post_training.evaluation.tool_calls import load_eval_cases


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="检查扩充 SFT 数据的标签、隔离和真实 token 长度")
    parser.add_argument("--config", type=Path, default=root / "configs/sft_lora_expanded_cpu.json")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    train = load_examples(root / config["train_data"])
    dev = load_examples(root / config["eval_data"])
    targets = load_eval_cases(root / config.get("validation_targets", "data/training/expanded_v1_targets.jsonl"))
    protected = load_eval_cases(root / "data/eval/tool_calls_v1.jsonl")
    protected += [target_for_example(item) for item in load_examples(root / "data/examples/sft_eval_en.jsonl")]
    summary = validate_dataset(train, dev, targets, protected)
    if not args.skip_tokenizer:
        tokenizer = load_hf_tokenizer(
            config["model"], revision=config["revision"], cache_dir=root / ".cache/huggingface",
            local_files_only=True,
        )
        for name, examples in (("train", train), ("development", dev)):
            encoded = [encode_sft(example, tokenizer, max_length=config["max_length"]) for example in examples]
            summary[name]["max_tokens"] = max(len(item["input_ids"]) for item in encoded)
            summary[name]["min_response_tokens"] = min(sum(item["loss_mask"]) for item in encoded)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("检查通过：标签合法、参考结果一致，训练/开发/原评测输入与精确表达式没有跨集重叠。")


if __name__ == "__main__":
    main()
