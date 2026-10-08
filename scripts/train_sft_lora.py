"""CPU smoke experiment: train, save, reload and check LoRA."""

import argparse
import gc
import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from agentic_post_training.data import collate_sft, encode_sft, load_examples
from agentic_post_training.data.hf_tokenizer import load_hf_tokenizer
from agentic_post_training.trainers.sft_lora import (
    attach_lora, check_split, evaluate_loss, model_inputs, train_steps,
)


@torch.no_grad()
def generate_answer(model, tokenizer, example, max_new_tokens):
    model.eval()
    ids = tokenizer.apply_chat_template([
        {"role": "system", "content": example.system},
        {"role": "user", "content": example.prompt},
    ], tokenize=True, add_generation_prompt=True)
    inputs = torch.tensor([ids], dtype=torch.long)
    output = model.generate(
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0, len(ids):].tolist())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, set_seed

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CPU 上验证 LoRA 训练、保存与重载")
    parser.add_argument("--config", type=Path, default=root / "configs/sft_lora_cpu.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config["cpu_threads"] <= 0 or config["max_new_tokens"] <= 0:
        raise ValueError("cpu_threads and max_new_tokens must be positive")
    set_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    cache = root / ".cache/huggingface"
    train = load_examples(root / config["train_data"])
    evaluation = load_examples(root / config["eval_data"])
    check_split(train, evaluation)
    tokenizer = load_hf_tokenizer(
        config["model"], revision=config["revision"], cache_dir=cache,
        local_files_only=args.local_files_only,
    )
    # Pin the base weights to the tokenizer snapshot when a Hub commit is available.
    revision = tokenizer.raw.init_kwargs.get("_commit_hash") or config["revision"]
    train_encoded = [encode_sft(item, tokenizer, max_length=config["max_length"]) for item in train]
    eval_encoded = [encode_sft(item, tokenizer, max_length=config["max_length"]) for item in evaluation]
    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            config["model"], revision=revision, cache_dir=str(cache),
            local_files_only=args.local_files_only, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager",
        ).cpu()

    print("加载模型权重；CPU float32，首次运行需要下载。", flush=True)
    base = load_base()
    revision = getattr(base.config, "_commit_hash", None) or revision
    model = attach_lora(
        base, rank=config["lora_rank"], alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
    )
    model.print_trainable_parameters()
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    output = root / "artifacts/checkpoints" / ("sft-lora-" + run_name)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "train_log.jsonl").open("w", encoding="utf-8") as log:
        def record_step(record):
            line = json.dumps(record)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        before_loss = evaluate_loss(model, eval_encoded, pad_token_id=tokenizer.pad_token_id)
        before_text = generate_answer(model, tokenizer, evaluation[0], config["max_new_tokens"])
        history = train_steps(
            model, train_encoded, pad_token_id=tokenizer.pad_token_id,
            max_steps=config["max_steps"], batch_size=config["batch_size"],
            learning_rate=config["learning_rate"], max_grad_norm=config["max_grad_norm"],
            on_step=record_step,
        )
    after_loss = evaluate_loss(model, eval_encoded, pad_token_id=tokenizer.pad_token_id)
    after_text = generate_answer(model, tokenizer, evaluation[0], config["max_new_tokens"])
    probe = model_inputs(collate_sft([eval_encoded[0]], pad_token_id=tokenizer.pad_token_id))
    model.eval()
    with torch.no_grad():
        expected_logits = model(**probe).logits.detach().clone()
    model.save_pretrained(output / "adapter")
    tokenizer.raw.save_pretrained(output / "tokenizer")
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    # Release model and optimizer graph before loading a fresh base on CPU.
    del model, base
    gc.collect()
    reloaded = PeftModel.from_pretrained(load_base(), str(output / "adapter"), is_trainable=False)
    reloaded.eval()
    with torch.no_grad():
        actual_logits = reloaded(**probe).logits
    if not torch.allclose(expected_logits, actual_logits, atol=1e-5, rtol=1e-5):
        raise ValueError("adapter reload logits do not match saved model")
    summary = {
        "config": config, "resolved_revision": revision,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "trainable_parameters": trainable_count, "total_parameters": total_count,
        "train_examples": len(train), "eval_examples": len(evaluation),
        "eval_loss_before": before_loss, "eval_loss_after": after_loss,
        "generation_before": before_text, "generation_after": after_text,
        "reload_logits_match": True,
        "reload_max_abs_error": (expected_logits - actual_logits).abs().max().item(),
        "steps_completed": len(history),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("实验产物：", output, flush=True)


if __name__ == "__main__":
    main()
