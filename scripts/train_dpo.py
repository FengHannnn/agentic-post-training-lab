"""Audit data, continue v2 with DPO, validate frozen weights and save/reload."""

import argparse
import gc
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from build_dpo_preferences import protected_cases
from freeze_sft_baseline import verify
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preferences, validate_preferences
from agentic_post_training.trainers.dpo import (
    cache_reference, encode_pair, evaluate_preferences, parameter_digest, train_dpo,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="CPU DPO 训练与保存重载验证")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dpo_cpu.json")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    config = json.loads(args.config.read_text(encoding="utf-8"))
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

    def load_base():
        return AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()

    print("加载两份固定 base＋v2 adapter，policy 可训练，reference 全冻结。", flush=True)
    policy = PeftModel.from_pretrained(load_base(), str(baseline / "checkpoint/adapter"), is_trainable=True)
    reference = PeftModel.from_pretrained(load_base(), str(baseline / "checkpoint/adapter"), is_trainable=False)
    reference.requires_grad_(False)
    reference.eval()
    policy.config.use_cache = False
    reference.config.use_cache = False
    initial_digest = parameter_digest(policy)
    reference_digest = parameter_digest(reference)
    if initial_digest != reference_digest:
        raise ValueError("policy/reference did not start from identical parameters")
    frozen_digest = parameter_digest(policy, frozen_only=True)
    print("初始参数一致；缓存 train/development 的固定 reference 分数。", flush=True)
    caches = {split: cache_reference(reference, selected, tokenizer.pad_token_id) for split, selected in pairs.items()}
    before = {split: evaluate_preferences(policy, selected, caches[split], tokenizer.pad_token_id, config["beta"])
              for split, selected in pairs.items()}
    if any(abs(value["relative_margin"]) > 1e-5 for value in before.values()):
        raise ValueError("policy/reference initial relative margin must be zero")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = ROOT / "artifacts/checkpoints" / f"dpo-lora-{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"before": before}, ensure_ascii=False), flush=True)
    with (output / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as log:
        def record(row):
            row["pair_id"] = samples["train"][row["pair_index"]].id
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
        history = train_dpo(policy, pairs["train"], caches["train"], tokenizer.pad_token_id,
            max_steps=config["max_steps"], learning_rate=config["learning_rate"], beta=config["beta"],
            max_grad_norm=config["max_grad_norm"], seed=config["seed"], on_step=record)
    print("更新完成；评测 train/development 并检查冻结参数。", flush=True)
    after = {split: evaluate_preferences(policy, selected, caches[split], tokenizer.pad_token_id, config["beta"])
             for split, selected in pairs.items()}
    reference_unchanged = parameter_digest(reference) == reference_digest
    reference_no_grad = all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters())
    frozen_unchanged = parameter_digest(policy, frozen_only=True) == frozen_digest
    policy_updated = parameter_digest(policy) != initial_digest
    if not all((reference_unchanged, reference_no_grad, frozen_unchanged, policy_updated)):
        raise ValueError("DPO parameter/gradient invariants failed")
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
    del policy, reference
    gc.collect()
    restored = PeftModel.from_pretrained(load_base(), str(output / "adapter"), is_trainable=False)
    restored.eval()
    with torch.no_grad():
        actual = restored(**inputs, use_cache=False).logits
    match = torch.allclose(expected, actual, atol=1e-5, rtol=1e-5)
    if not match:
        raise ValueError("DPO reload logits mismatch")
    result = {
        "config": config, "model": manifest["model"], "resolved_revision": manifest["resolved_revision"],
        "baseline_manifest_sha256": baseline_hash, "data_manifest": data_manifest, "audit": audit,
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        "before": before, "after": after, "trainable_parameters": trainable_count,
        "steps_completed": len(history), "reference_unchanged": reference_unchanged,
        "reference_no_grad": reference_no_grad, "policy_frozen_parameters_unchanged": frozen_unchanged,
        "policy_updated": policy_updated, "initial_policy_reference_match": True,
        "reload_logits_match": bool(match), "reload_max_abs_error": (expected - actual).abs().max().item(),
        "reference_cache": "fixed v2 response-only summed log probabilities; detached; no sampling",
        "checkpoint_scope": "adapter and tokenizer; optimizer state not saved",
    }
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"after": after, "reference_unchanged": reference_unchanged,
                      "policy_updated": policy_updated, "reload_logits_match": bool(match),
                      "reload_max_abs_error": result["reload_max_abs_error"], "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
