"""Check a real tokenizer against the existing SFT preprocessing."""

import argparse
import sys
from pathlib import Path

from agentic_post_training.data import collate_sft, encode_sft, load_examples
from agentic_post_training.data.hf_tokenizer import DEFAULT_TOKENIZER, load_hf_tokenizer


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="检查真实 tokenizer、回答边界与特殊 token")
    parser.add_argument("--model", default=DEFAULT_TOKENIZER)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--data", type=Path, default=root / "data/examples/sft_train.jsonl")
    parser.add_argument("--cache-dir", type=Path, default=root / ".cache/huggingface")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    tokenizer = load_hf_tokenizer(
        args.model, cache_dir=args.cache_dir, revision=args.revision,
        local_files_only=args.local_files_only,
    )
    raw = tokenizer.raw
    print("Tokenizer：", args.model)
    print("词表大小（含新增 token）：", len(raw))
    print("PAD：", repr(raw.pad_token), raw.pad_token_id)
    print("EOS：", repr(raw.eos_token), raw.eos_token_id)
    print("PAD 与 EOS 同编号：", raw.pad_token_id == raw.eos_token_id)
    examples = load_examples(args.data)
    encoded = []
    for example in examples:
        item = encode_sft(example, tokenizer, max_length=args.max_length)
        boundary = item["loss_mask"].index(1)
        response_ids = item["input_ids"][boundary:]
        decoded = tokenizer.decode(response_ids)
        if not decoded.startswith(example.response):
            raise ValueError(f"{example.id}: decoded response does not match original text")
        if raw.eos_token_id is not None and raw.eos_token_id not in response_ids:
            raise ValueError(f"{example.id}: response contains no tokenizer EOS; inspect template")
        encoded.append(item)
        print(f"\n样本 {example.id}：总长度={len(item['input_ids'])}，回答起点={boundary}")
        print("prompt 尾部：", repr(tokenizer.decode(item["input_ids"][:boundary])[-100:]))
        print("参与训练的回答：", repr(decoded))
        print("最后 8 个 token：", raw.convert_ids_to_tokens(response_ids[-8:]))
    batch = collate_sft(encoded, pad_token_id=tokenizer.pad_token_id)
    padding = batch["attention_mask"] == 0
    if not (batch["labels"][padding] == -100).all().item():
        raise ValueError("padding labels must be ignored")
    if not (batch["loss_mask"][padding] == 0).all().item():
        raise ValueError("padding must not contribute to loss")
    print("\nbatch shape：", tuple(batch["input_ids"].shape))
    print("有效长度：", batch["attention_mask"].sum(dim=1).tolist())
    print("训练 token 数：", batch["loss_mask"].sum(dim=1).tolist())
    print("检查通过：前缀匹配、回答文本、EOS 和 padding 标签均符合当前约定。")


if __name__ == "__main__":
    main()
