"""CPU-only algebra audit of token TIS on already-scored real rollout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import pack_trainer_scores


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "artifacts/evaluations/rollout-correction-contract-v1")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config_path = ROOT / "configs/rollout_correction_contract.json"
    config = read(config_path)
    source = ROOT / "artifacts/evaluations/trainer-probability-contract-v1/summary.json"
    if sha(source) != config["trainer_summary_sha256"]:
        raise ValueError("trainer probability result differs from bound input")
    if config != {"contract_version": "token-tis-policy-surrogate-v1", "aggregation": "token",
        "upper_cap": 2.0, "clip_epsilon": .2, "weight_normalization": "none",
        "loss_reduction": "eligible_token_mean", "beta": 0.0, "stop_gradient_weights": True,
        "unbiased_sequence_or_group_correction_claimed": False,
        "trainer_summary_sha256": sha(source), "training_admitted": False}:
        raise ValueError("unsupported correction contract")
    report = read(source)
    rollout = ROOT / "artifacts/trajectories/rollout-vllm-phase4-20261020-01"
    for name, digest in report["input_sha256"].items():
        if sha(rollout / name) != digest:
            raise ValueError(f"rollout input changed: {name}")
    for name, digest in report["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"trainer source changed: {name}")
    if (not report["same_path_no_update_ratio_passed"] or not report["parameters_unchanged"]
        or report["max_old_new_logp_delta"] != 0 or report["training_admitted"]):
        raise ValueError("expected the completed no-update contract")
    originals = [json.loads(line) for line in (rollout / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = {}
    original_index = {}
    for row in originals:
        sample = RolloutSample(**row["sample"])
        grouped.setdefault(sample.group_id, []).append(sample)
        key = sample.group_id, sample.sample_index
        if key in original_index:
            raise ValueError("duplicate original sample")
        original_index[key] = sample
    rewards = read(rollout / "reward_audit.json")
    reward_index = {(r["group_id"], r["sample_index"]): r for r in rewards["records"]}
    for row in report["records"]:
        key = row["group_id"], row["sample_index"]
        if (row["sampler_behavior_token_log_probs"] != original_index[key].behavior_token_log_probs
            or row["trainer_old_token_log_probs"] != row["trainer_new_token_log_probs"]
            or any(row[k] != reward_index[key][k] for k in ("eligible", "reward", "advantage"))):
            raise ValueError("trainer scores no longer match original rollout/reward")
    groups = list(grouped.values())
    # Padding identity is determined by lengths, so this placeholder ID is safe.
    batch = pack_trainer_scores(groups, report["records"], pad_token_id=0)
    adv = torch.tensor([[reward_index[g[0].group_id, i]["advantage"] for i in range(len(g))]
                        for g in groups], dtype=torch.float32)
    # A score tensor leaf exercises the algebra only; it is NOT a model forward.
    new = batch["trainer_new_log_probs"].clone().requires_grad_()
    loss, stats = corrected_grpo_policy_loss(new, batch["trainer_old_log_probs"],
        batch["sampler_behavior_log_probs"], adv, batch["eligible_response_mask"],
        upper_cap=config["upper_cap"], clip_epsilon=config["clip_epsilon"])
    score_gradient, = torch.autograd.grad(loss, new)
    passed = bool(torch.isfinite(loss) and loss.item() == 0 and (score_gradient == 0).all())
    result = {"contract": config, "records": len(report["records"]),
        "eligible_tokens": int(batch["eligible_response_mask"].sum()),
        "correction_weights": stats, "corrected_policy_loss": loss.item(),
        "max_abs_score_gradient": score_gradient.abs().max().item(),
        "zero_advantage_expectation_passed": passed,
        "has_reward_learning_signal": rewards["has_reward_learning_signal"],
        "original_cross_engine_gate_passed": False, "model_loaded": False,
        "score_tensor_autograd_checked": True, "model_backward_calls": 0,
        "new_generation_calls": 0, "optimizer_steps": 0, "training_admitted": False,
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (
            Path(__file__), config_path, ROOT / "src/agentic_post_training/algorithms/rollout_correction.py",
            ROOT / "src/agentic_post_training/rollouts/trainer_scores.py")},
        "input_summary_sha256": sha(source), "input_rollout_sha256": report["input_sha256"]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
