"""No-update policy ratio check on existing rollout, with no optimizer or backward."""

import argparse
import json
import os
from pathlib import Path

from collect_vllm_rollout import ROOT, audit, context, load_and_validate, check_gpu
from verify_cloud_bundle import file_hash
from agentic_post_training.rollouts.trainer_scores import (
    score_response, no_update_metrics, sampler_trainer_difference, pack_trainer_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
        default=ROOT / "artifacts/trajectories/rollout-vllm-phase4-20261020-01")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "artifacts/evaluations/trainer-probability-contract-v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    config_path = ROOT / "configs/trainer_probability_contract.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    diagnostic_path = ROOT / "artifacts/evaluations/rollout-numeric-diagnostic/diagnostic.json"
    if (file_hash(diagnostic_path) != config["diagnostic_sha256"]
        or file_hash(args.input_dir / "run_manifest.json") != config["rollout_manifest_sha256"]):
        raise ValueError("bound diagnostic or rollout identity differs")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if not diagnostic["diagnosis_completed"] or diagnostic["new_generation_calls"] != 0:
        raise ValueError("finite numerical diagnostic must be complete")
    if (config["dtype"] != "float32" or config["attention"] != "eager" or config["use_cache"]
        or config["tf32"] or config["behavior_correction_applied"] or config["training_updates"] != 0
        or config["absolute_old_new_logp_tolerance"] != 1e-5):
        raise ValueError("unsupported trainer scoring contract")
    audit(args.input_dir)
    rollout_config, baseline, base, tokenizer, rows, groups, sampler_batch, rewards = load_and_validate(args.input_dir)
    device = check_gpu(rollout_config)
    if device != diagnostic["device"]:
        raise ValueError("same GPU model required for this contract check")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from agentic_post_training.trainers.dpo import parameter_digest
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
        trust_remote_code=False, dtype=torch.float32, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(baseline / "checkpoint/adapter"),
        is_trainable=True, autocast_adapter_dtype=False).to("cuda").eval()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if (not trainable or any("lora_" not in name for name in trainable)
        or count != config["expected_trainable_parameters"]
        or any(parameter.dtype != torch.float32 for parameter in model.parameters())
        or any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in model.modules())):
        raise ValueError("expected only fixed-rank FP32 LoRA trainable, dropout zero")
    initial_digest = parameter_digest(model)
    if initial_digest != config["frozen_fp32_parameter_sha256"]:
        raise ValueError("trainer initialization differs from original FP32 SFT parameters")
    records = []
    indexed_rewards = {(r["group_id"], r["sample_index"]): r for r in rewards["records"]}
    for group in groups:
        for sample in group:
            old = score_response(model, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=False)
            new = score_response(model, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=True)
            metrics = no_update_metrics(old, new, config["absolute_old_new_logp_tolerance"])
            reward = indexed_rewards[sample.group_id, sample.sample_index]
            record = {"group_id": sample.group_id, "sample_index": sample.sample_index,
                "response_token_ids": sample.response_token_ids,
                "trainer_old_token_log_probs": old.cpu().tolist(),
                "trainer_new_token_log_probs": new.detach().cpu().tolist(),
                "sampler_behavior_token_log_probs": sample.behavior_token_log_probs,
                "eligible": reward["eligible"], "reward": reward["reward"], "advantage": reward["advantage"],
                "no_update_check": metrics,
                "sampler_trainer_difference": sampler_trainer_difference(sample.behavior_token_log_probs, old.cpu().tolist())}
            records.append(record)
            del old, new  # Free this response's graph; no backward is performed.
    final_digest = parameter_digest(model)
    no_grad_accumulated = all(parameter.grad is None for parameter in model.parameters())
    batch = pack_trainer_scores(groups, records, tokenizer.pad_token_id)
    passed = (all(row["no_update_check"]["passed"] and row["no_update_check"]["ratio_requires_grad"]
                  for row in records) and initial_digest == final_digest and no_grad_accumulated)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(batch, args.output_dir / "trainer_score_batch.pt")
    report = {"config": config, "records": records, "versions": rollout_config["versions"],
        "device": device, "trainable_parameters": count, "trainable_parameter_names": trainable,
        "initial_parameter_sha256": initial_digest, "final_parameter_sha256": final_digest,
        "no_grad_accumulated": no_grad_accumulated, "parameters_unchanged": initial_digest == final_digest,
        "same_path_no_update_ratio_passed": passed,
        "max_old_new_logp_delta": max(r["no_update_check"]["max_abs_logp_delta"] for r in records),
        "max_ratio_minus_one": max(r["no_update_check"]["max_abs_ratio_minus_one"] for r in records),
        "sampler_trainer_max_logp_difference": max(r["sampler_trainer_difference"]["max_abs_logp_delta"] for r in records),
        "eligible_samples": rewards["eligible_samples"], "zero_variance_rate": rewards["zero_variance_rate"],
        "has_reward_learning_signal": rewards["has_reward_learning_signal"],
        "batch_shape": list(batch["response_mask"].shape), "response_tokens": int(batch["response_mask"].sum()),
        "eligible_response_tokens": int(batch["eligible_response_mask"].sum()),
        "new_generation_calls": 0, "backward_calls": 0, "optimizer_steps": 0,
        "behavior_correction_applied": False, "training_admitted": False,
        "original_cross_engine_gate_passed": False,
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in (
            Path(__file__), config_path, ROOT / "src/agentic_post_training/rollouts/trainer_scores.py",
            ROOT / "src/agentic_post_training/algorithms/losses.py", ROOT / "src/agentic_post_training/trainers/dpo.py")},
        "input_sha256": {name: file_hash(args.input_dir / name) for name in (
            "run_manifest.json", "rollouts.jsonl", "replay_checks.json", "reward_audit.json")},
        "diagnostic_sha256": config["diagnostic_sha256"],
        "batch_sha256": file_hash(args.output_dir / "trainer_score_batch.pt")}
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("same_path_no_update_ratio_passed", "parameters_unchanged",
        "trainable_parameters", "no_grad_accumulated", "max_old_new_logp_delta", "max_ratio_minus_one",
        "sampler_trainer_max_logp_difference", "zero_variance_rate", "new_generation_calls",
        "backward_calls", "optimizer_steps", "training_admitted")}, ensure_ascii=False))
    print(f"Saved: {args.output_dir / 'summary.json'}", flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
