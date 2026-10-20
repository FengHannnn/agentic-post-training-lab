"""Score four existing sequences through controlled HF numerical paths; no generation."""

import argparse
import json
import os
from pathlib import Path

from collect_vllm_rollout import ROOT, audit, context, check_gpu
from replay_cloud_smoke import compare_logps, response_logps
from verify_cloud_bundle import file_hash


def full_logps(model, prompt, response):
    import torch
    if not prompt or not response:
        raise ValueError("non-empty native prompt and response required")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt + response], device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        return response_logps(logits, ids, len(prompt)).cpu().tolist()


def cached_logps(model, prompt, response):
    import torch
    if not prompt or not response:
        raise ValueError("non-empty native prompt and response required")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], device=device)
    values = []
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
        for index, token in enumerate(response):
            values.append(torch.log_softmax(output.logits[0, -1].float(), -1)[token].item())
            if index + 1 < len(response):
                # Feed the recorded token, not argmax or a newly sampled token.
                output = model(input_ids=torch.tensor([[token]], device=device),
                    attention_mask=torch.ones((1, len(prompt) + index + 1), dtype=torch.long, device=device),
                    past_key_values=output.past_key_values, use_cache=True)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
        default=ROOT / "artifacts/trajectories/rollout-vllm-phase4-20261020-01")
    parser.add_argument("--output", type=Path,
        default=ROOT / "artifacts/evaluations/rollout-numeric-diagnostic/diagnostic.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    config_path = ROOT / "configs/rollout_numeric_diagnostic.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for name, expected in config["input_sha256"].items():
        if file_hash(args.input_dir / name) != expected:
            raise ValueError(f"original experiment changed: {name}")
    audit(args.input_dir)
    rollout_config, baseline, base, tokenizer, cases, prompts = context()
    device = check_gpu(rollout_config)
    manifest = json.loads((args.input_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if device != manifest["device"]:
        raise ValueError("diagnostic must run on the same GPU model")
    rows = [json.loads(line) for line in (args.input_dir / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    old_replay = json.loads((args.input_dir / "replay_checks.json").read_text(encoding="utf-8"))["records"]
    if config["selected_flat_indices"] != [0, 4, 9, 13] or config["original_absolute_logp_limit"] != 0.001:
        raise ValueError("diagnostic scope or original limit changed")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float16, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(baseline / "checkpoint/adapter"),
        is_trainable=False, autocast_adapter_dtype=False).to("cuda").eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("diagnostic model must be frozen")
    numerical_settings = {"fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "initial_matmul_precision": torch.get_float32_matmul_precision(),
        "initial_tf32": torch.backends.cuda.matmul.allow_tf32}
    results = []
    for flat in config["selected_flat_indices"]:
        sample = rows[flat]["sample"]
        prompt, response = sample["prompt_token_ids"], sample["response_token_ids"]
        result = {"flat_index": flat, "case_id": sample["case_id"], "sample_index": sample["sample_index"],
            "prompt_token_ids": prompt, "response_token_ids": response,
            "vllm_fp16_logps": sample["policy_token_log_probs"],
            "saved_hf_fp16_full_logps": old_replay[flat]["replayed_policy_log_probs"],
            "hf_fp16_full_logps": full_logps(model, prompt, response),
            "hf_fp16_cached_logps": cached_logps(model, prompt, response)}
        results.append(result)
    # Preserve the already FP16-rounded weights, changing compute precision only.
    model.float()
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("FP32 diagnostic parameter conversion incomplete")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for result in results:
        result["hf_fp32_rounded_weights_logps"] = full_logps(model,
            result["prompt_token_ids"], result["response_token_ids"])
        pairs = {"fp16_full_repeat": ("saved_hf_fp16_full_logps", "hf_fp16_full_logps"),
            "vllm_vs_hf_fp16_full": ("vllm_fp16_logps", "hf_fp16_full_logps"),
            "vllm_vs_hf_fp16_cached": ("vllm_fp16_logps", "hf_fp16_cached_logps"),
            "hf_fp16_full_vs_cached": ("hf_fp16_full_logps", "hf_fp16_cached_logps"),
            "hf_fp16_vs_fp32_compute": ("hf_fp16_full_logps", "hf_fp32_rounded_weights_logps"),
            "vllm_vs_hf_fp32_compute": ("vllm_fp16_logps", "hf_fp32_rounded_weights_logps")}
        result["comparisons"] = {name: compare_logps(result[a], result[b], 0.001)
                                 for name, (a, b) in pairs.items()}
    report = {"config": config, "device": device, "versions": rollout_config["versions"],
        "results": results, "diagnosis_completed": True, "new_generation_calls": 0,
        "training_updates": 0, "training_admitted": False, "original_rollout_gate_passed": False,
        "fp32_weight_origin": "FP16 model cast to FP32; not reloaded from original FP32 adapter",
        "fp32_tf32_enabled": False, "initial_numerical_settings": numerical_settings,
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in (
            Path(__file__), config_path, ROOT / "scripts/collect_vllm_rollout.py",
            ROOT / "scripts/replay_cloud_smoke.py")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"diagnosis_completed": True, "new_generation_calls": 0,
        "max_errors": [{"case_id": row["case_id"], "sample_index": row["sample_index"],
            **{name: value["max_abs_error"] for name, value in row["comparisons"].items()}}
            for row in results], "output": str(args.output), "training_admitted": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
