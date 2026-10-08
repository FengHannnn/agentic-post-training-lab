"""Inspect response masks offline without downloading a model."""

import argparse
from pathlib import Path

from agentic_post_training.data import collate_sft, encode_sft, load_examples
from agentic_post_training.data.demo_tokenizer import DemoChatTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="离线检查 SFT 回答边界与 padding")
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/examples/sft_train.jsonl",
    )
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    tokenizer = DemoChatTokenizer()
    encoded = []
    print("使用教学字符 tokenizer；不是预训练模型 tokenizer。")
    for example in load_examples(args.data):
        item = encode_sft(example, tokenizer, max_length=args.max_length)
        encoded.append(item)
        boundary = item["loss_mask"].index(1)
        print(f"\n样本 {example.id}：总长度={len(item['input_ids'])}，回答起点={boundary}")
        print("参与训练的文本：", tokenizer.decode(item["input_ids"][boundary:]))
    batch = collate_sft(encoded, pad_token_id=tokenizer.pad_token_id)
    print("batch shape：", tuple(batch["input_ids"].shape))
    print("每条样本有效长度：", batch["attention_mask"].sum(dim=1).tolist())
    print("每条样本训练 token 数：", batch["loss_mask"].sum(dim=1).tolist())


if __name__ == "__main__":
    main()
