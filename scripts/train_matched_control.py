"""Execute the frozen three-arm plan once, reusing tested training loops."""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, write_json
from freeze_sft_baseline import verify
from matched_control_common import ARMS, DATA, ROOT, PLAN_HASH, exposure_order, load_design, read_json, read_rows, verify_exposures
from agentic_post_training.data import collate_sft, encode_sft
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.trainers.dpo import cache_reference, encode_pair, evaluate_preferences, objective, parameter_digest, train_dpo
from agentic_post_training.trainers.sft_lora import evaluate_loss, train_steps


def cached_metrics(values, beta):
    rows = []
    for value in values:
        output = objective(value, value, beta)
        rows.append({"loss": output.loss.item(), "relative_margin": output.logits.item(),
            "chosen_log_prob": value[0].item(), "rejected_log_prob": value[1].item()})
    return {key: sum(row[key] for row in rows)/len(rows) for key in rows[0]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="执行冻结的38步三组训练，不扫描超参数")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.output is None:
        parser.error("resume requires --output")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import PeftModel
    subprocess.run([sys.executable, str(ROOT / "scripts/build_matched_preferences.py"), "--audit-only"], check=True)
    plan, data_manifest, baseline, samples = load_design()
    verify(baseline)
    manifest = read_json(baseline / "manifest.json")
    config = plan["common"]
    torch.set_num_threads(config["cpu_threads"])
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"), local_files_only=True, trust_remote_code=False))
    load_design(tokenizer)
    pairs = {label: [encode_pair(row, tokenizer, config["max_length"]) for row in samples[label]] for label in ARMS[1:]}
    chosen = [encode_sft(row, tokenizer, max_length=config["max_length"]) for row in samples["chosen_sft"]]
    order = exposure_order(len(chosen), config["seed"])
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else ROOT / "artifacts/checkpoints" / f"matched-control-{stamp}-{uuid4().hex[:8]}"
    sources = [Path(__file__), ROOT / "scripts/matched_control_common.py",
        ROOT / "src/agentic_post_training/trainers/dpo.py", ROOT / "src/agentic_post_training/trainers/sft_lora.py",
        ROOT / "src/agentic_post_training/data/sft.py", ROOT / "src/agentic_post_training/algorithms/losses.py"]
    inventory = {"plan_sha256": PLAN_HASH, "data_manifest_sha256": file_hash(DATA / "manifest.json"),
        "baseline_manifest_sha256": plan["baseline_manifest_sha256"], "model": manifest["model"],
        "resolved_revision": manifest["resolved_revision"], "plan": plan,
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
        "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")}}
    if args.resume:
        if read_json(output / "run_manifest.json") != inventory:
            raise ValueError("resume inventory differs")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", inventory)
    print(json.dumps({"output": str(output), "plan_sha256": PLAN_HASH}, ensure_ascii=False), flush=True)

    def load_model(trainable):
        base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()
        model = PeftModel.from_pretrained(base, str(baseline / "checkpoint/adapter"), is_trainable=trainable)
        if not trainable:
            model.requires_grad_(False)
        model.config.use_cache = False
        return model

    reference = load_model(False)
    reference.eval()
    reference_digest = parameter_digest(reference)
    cache_file = output / "reference_log_probs.pt"
    if args.resume and cache_file.exists():
        # No cached tensor is reused without its original hash and full identity.
        reference_manifest = read_json(output / "reference_manifest.json")
        if file_hash(cache_file) != reference_manifest["sha256"] or reference_manifest["parameter_sha256"] != reference_digest:
            raise ValueError("reference cache changed")
        caches = torch.load(cache_file, map_location="cpu", weights_only=True)
    else:
        caches = {label: cache_reference(reference, selected, tokenizer.pad_token_id) for label, selected in pairs.items()}
        torch.save(caches, cache_file)
        write_json(output / "reference_manifest.json", {"sha256": file_hash(cache_file), "parameter_sha256": reference_digest})
    if set(caches) != set(ARMS[1:]) or any(len(caches[label]) != len(pairs[label]) or any(
        value.shape != (2,) or value.requires_grad or not torch.isfinite(value).all().item() for value in caches[label]) for label in pairs):
        raise ValueError("invalid reference cache")
    from agentic_post_training.trainers.dpo import pair_log_probs
    with torch.no_grad():
        for label in pairs:
            if not torch.allclose(pair_log_probs(reference, pairs[label][0], tokenizer.pad_token_id), caches[label][0], atol=1e-6, rtol=1e-6):
                raise ValueError("reference cache fresh probe differs")
    results, histories = {}, {}
    for label in ARMS:
        folder = output / label
        if (folder / "summary.json").exists():
            saved = read_json(folder / "summary.json")
            if (not args.resume or saved["adapter_sha256"] != file_hash(folder / "adapter/adapter_model.safetensors")
                or saved["initial_policy_sha256"] != reference_digest or saved["plan_sha256"] != PLAN_HASH
                or saved["train_log_sha256"] != file_hash(folder / "train_log.jsonl")):
                raise ValueError("completed arm differs")
            histories[label] = read_rows(folder / "train_log.jsonl")
            results[label] = saved
            continue
        if folder.exists():
            raise ValueError("partial training arm cannot be rerun under the frozen budget")
        folder.mkdir()
        set_seed(config["seed"])
        policy = load_model(True)
        initial = parameter_digest(policy)
        frozen = parameter_digest(policy, frozen_only=True)
        if initial != reference_digest or any("lora_" not in name for name, value in policy.named_parameters() if value.requires_grad):
            raise ValueError("initial model or trainable parameter scope differs")
        beta = 0.1
        before = cached_metrics(caches[label], beta) if label != "chosen_sft" else {
            "chosen_token_mean_nll": -sum(value[0].item() for value in caches["real_dpo"])/sum(sum(row["loss_mask"][1:]) for row in chosen)}
        print(f"start {label}: 38 steps", flush=True)
        with (folder / "train_log.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            def record(row):
                index = order[row["step"]-1]
                if "pair_index" in row and row["pair_index"] != index:
                    raise ValueError("trainer sample order differs")
                selected = [chosen[index]] if label == "chosen_sft" else pairs[label][index]
                row.update(pair_index=index, pair_id=samples["chosen_sft"][index].id,
                    chosen_response_tokens=sum(selected[0]["loss_mask"][1:]),
                    rejected_response_tokens=0 if label == "chosen_sft" else sum(selected[1]["loss_mask"][1:]),
                    input_tokens=sum(len(value["input_ids"]) for value in selected),
                    padded_input_tokens=len(selected)*max(len(value["input_ids"]) for value in selected))
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                print(json.dumps({"arm": label, **row}), flush=True)
            started = time.perf_counter()
            if label == "chosen_sft":
                history = train_steps(policy, [chosen[index] for index in order], pad_token_id=tokenizer.pad_token_id,
                    max_steps=config["max_steps"], batch_size=1, learning_rate=config["learning_rate"],
                    max_grad_norm=config["max_grad_norm"], on_step=record)
            else:
                history = train_dpo(policy, pairs[label], caches[label], tokenizer.pad_token_id,
                    max_steps=config["max_steps"], learning_rate=config["learning_rate"], beta=plan["arms"][label]["beta"],
                    max_grad_norm=config["max_grad_norm"], seed=config["seed"], on_step=record, chosen_nll_weight=0.0)
            seconds = time.perf_counter()-started
        after = {"chosen_token_mean_nll": evaluate_loss(policy, chosen, pad_token_id=tokenizer.pad_token_id)} if label == "chosen_sft" else evaluate_preferences(policy, pairs[label], caches[label], tokenizer.pad_token_id, beta)
        invariants = {"policy_frozen_parameters_unchanged": parameter_digest(policy, frozen_only=True) == frozen,
            "policy_updated": parameter_digest(policy) != initial, "reference_unchanged": parameter_digest(reference) == reference_digest,
            "reference_no_grad": all(not value.requires_grad and value.grad is None for value in reference.parameters())}
        if not all(invariants.values()):
            raise ValueError("training invariants failed")
        probe = collate_sft(encode_pair(samples["development"][0], tokenizer, config["max_length"]), pad_token_id=tokenizer.pad_token_id)
        inputs = {key: probe[key] for key in ("input_ids", "attention_mask")}
        policy.eval()
        with torch.no_grad():
            expected = policy(**inputs, use_cache=False).logits.detach().clone()
        policy.save_pretrained(folder / "adapter", save_embedding_layers=False)
        tokenizer.raw.save_pretrained(folder / "tokenizer")
        del policy
        gc.collect()
        from peft import PeftModel
        restored_base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()
        restored = PeftModel.from_pretrained(restored_base, str(folder / "adapter"), is_trainable=False)
        restored.eval()
        with torch.no_grad():
            actual = restored(**inputs, use_cache=False).logits
        if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
            raise ValueError("reload logits differ")
        results[label] = {"arm": label, "plan_sha256": PLAN_HASH, "initial_policy_sha256": initial,
            "initial_reference_sha256": reference_digest, "before": before, "after": after,
            "before_origin": "derived from frozen reference scores; identical initial policy parameter digest",
            "steps_completed": len(history), "training_seconds": seconds, **invariants,
            "reload_logits_match": True, "reload_max_abs_error": (expected-actual).abs().max().item(),
            "adapter_sha256": file_hash(folder / "adapter/adapter_model.safetensors"),
            "train_log_sha256": file_hash(folder / "train_log.jsonl"),
            "tokens": {key: sum(row[key] for row in history) for key in
                ("chosen_response_tokens", "rejected_response_tokens", "input_tokens", "padded_input_tokens")},
            "optimizer_state_saved": False}
        write_json(folder / "summary.json", results[label])
        histories[label] = history
        print(json.dumps({"arm_completed": label, "summary": results[label]}, ensure_ascii=False), flush=True)
        del restored, restored_base, actual, expected
        gc.collect()
    verify_exposures(histories, samples["chosen_sft"], plan)
    result = {**inventory, "arms": results, "matching_exposure_verified": True,
        "reference_cache_sha256": file_hash(cache_file),
        "arm_summary_sha256": {label: file_hash(output / label / "summary.json") for label in ARMS},
        "baseline_replaced": False, "checkpoint_scope": "adapters/tokenizers/logs; no optimizer states"}
    write_json(output / "summary.json", result)
    print(json.dumps({"output": str(output), "all_three_arms_complete": True}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
