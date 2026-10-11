"""Chosen-only SFT control with the exact DPO starting point and exposure order."""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from build_dpo_preferences import protected_cases
from freeze_sft_baseline import verify
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, validate_preferences
from agentic_post_training.trainers.dpo import encode_pair, parameter_digest

from agentic_post_training.trainers.sft_lora import evaluate_loss, train_steps

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="chosen-only 继续 SFT 对照与保存重载验证")
    parser.add_argument("--dpo-run", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    dpo_run = args.dpo_run.resolve()
    dpo_summary = json.loads((dpo_run / "summary.json").read_text(encoding="utf-8"))
    config = dict(dpo_summary["config"])
    if config["max_steps"] != 28:
        raise ValueError("this control requires the original 28-step DPO run")
    if config["cpu_threads"] <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(config["cpu_threads"])
    set_seed(config["seed"])
    baseline = ROOT / config["baseline"]
    verify(baseline)
    manifest_path = baseline / "manifest.json"
    baseline_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = ROOT / config["preference_data"]
    data_manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if data_manifest["baseline_manifest_sha256"] != baseline_hash:
        raise ValueError("preference data and baseline manifest disagree")
    for name, digest in data_manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"preference data hash mismatch: {name}")
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    samples = {split: load_preferences(folder / f"{split}.jsonl") for split in ("train", "development")}
    audit = validate_preferences(samples["train"], samples["development"], protected_cases(),
                                 tokenizer, config["max_length"])
    pairs = {split: [encode_pair(row, tokenizer, config["max_length"]) for row in rows]
             for split, rows in samples.items()}

    chosen = {split: [pair[0] for pair in selected] for split, selected in pairs.items()}

    def load_base():
        return AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()

    print("从冻结 v2 adapter 继续 SFT；只学习 chosen。", flush=True)
    policy = PeftModel.from_pretrained(load_base(), str(baseline / "checkpoint/adapter"), is_trainable=True)
    policy.config.use_cache = False
    initial_digest = parameter_digest(policy)
    if baseline_hash != dpo_summary["baseline_manifest_sha256"] or data_manifest != dpo_summary["data_manifest"]:
        raise ValueError("control baseline/data differs from DPO")
    frozen_digest = parameter_digest(policy, frozen_only=True)
    if any("lora_" not in name for name, parameter in policy.named_parameters() if parameter.requires_grad):
        raise ValueError("control must train only LoRA parameters")
    before = {split: evaluate_loss(policy, selected, pad_token_id=tokenizer.pad_token_id)
              for split, selected in chosen.items()}
    # Match DPO's seeded randperm and each optimizer step's chosen exposure.
    order = torch.randperm(len(chosen["train"]), generator=torch.Generator().manual_seed(config["seed"])).tolist()
    dpo_history = [json.loads(line) for line in (dpo_run / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    if order != [row["pair_index"] for row in dpo_history] or len(order) != config["max_steps"]:
        raise ValueError("control order/exposures differ from DPO")
    ordered = [chosen["train"][index] for index in order]
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/checkpoints" / f"continued-sft-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"before": before}, ensure_ascii=False), flush=True)
    with (output / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        def record(row):
            row["pair_index"] = order[row["step"] - 1]
            row["pair_id"] = samples["train"][row["pair_index"]].id
            row["input_tokens"] = len(ordered[row["step"] - 1]["input_ids"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        started = time.perf_counter()
        history = train_steps(policy, ordered, pad_token_id=tokenizer.pad_token_id,
            max_steps=config["max_steps"], batch_size=1, learning_rate=config["learning_rate"],
            max_grad_norm=config["max_grad_norm"], on_step=record)
        training_seconds = time.perf_counter() - started
    after = {split: evaluate_loss(policy, selected, pad_token_id=tokenizer.pad_token_id)
             for split, selected in chosen.items()}
    frozen_unchanged = parameter_digest(policy, frozen_only=True) == frozen_digest
    policy_updated = parameter_digest(policy) != initial_digest
    if not frozen_unchanged or not policy_updated:
        raise ValueError("continued SFT parameter invariants failed")
    policy.eval()
    from agentic_post_training.data import collate_sft
    probe = collate_sft(pairs["development"][0], pad_token_id=tokenizer.pad_token_id)
    inputs = {name: probe[name] for name in ("input_ids", "attention_mask")}
    with torch.no_grad():
        expected = policy(**inputs, use_cache=False).logits.detach().clone()
    # q_proj/v_proj only; embeddings and vocabulary are unchanged and frozen.
    policy.save_pretrained(output / "adapter", save_embedding_layers=False)
    tokenizer.raw.save_pretrained(output / "tokenizer")
    print("adapter 已保存；释放模型并使用新 base 重载比较 logits。", flush=True)
    trainable_count = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    del policy
    gc.collect()
    restored = PeftModel.from_pretrained(load_base(), str(output / "adapter"), is_trainable=False)
    restored.eval()
    with torch.no_grad():
        actual = restored(**inputs, use_cache=False).logits
    match = torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
    if not match:
        raise ValueError("continued SFT reload logits mismatch")
    result = {
        "config": config, "model": manifest["model"], "resolved_revision": manifest["resolved_revision"],
        "baseline_manifest_sha256": baseline_hash, "data_manifest": data_manifest, "audit": audit,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "before": before, "after": after, "trainable_parameters": trainable_count,
        "steps_completed": len(history), "dpo_run": str(dpo_run), "training_seconds": training_seconds,
        "training_response_tokens": sum(row["response_tokens"] for row in history),
        "training_input_tokens": sum(row["input_tokens"] for row in history),
        "policy_frozen_parameters_unchanged": frozen_unchanged,
        "policy_updated": policy_updated, "pair_order_matches_dpo": True,
        "reload_logits_match": bool(match), "reload_max_abs_error": (expected - actual).abs().max().item(),
        "objective": "response-only token-mean cross entropy on chosen; no rejected",
        "checkpoint_scope": "adapter and tokenizer; optimizer state not saved",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"after": after, "dpo_run": str(dpo_run), "training_seconds": training_seconds,
        "training_response_tokens": sum(row["response_tokens"] for row in history),
        "training_input_tokens": sum(row["input_tokens"] for row in history),
                      "policy_updated": policy_updated, "reload_logits_match": bool(match),
                      "reload_max_abs_error": result["reload_max_abs_error"], "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
