"""Offline reward and group-signal audit of saved rollout records."""

from collections import defaultdict
from dataclasses import asdict

import torch

from agentic_post_training.algorithms.losses import group_relative_advantages
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rewards.contract import CONTRACT_VERSION, score_reward


def audit_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("empty rollout")
    groups = defaultdict(list)
    targets = {}
    results = []
    for record in records:
        sample = RolloutSample(**record["sample"])
        case = EvalCase(**record["case"])
        if sample.case_id != case.id or sample.group_id != case.id:
            raise ValueError("case/group identity mismatch")
        if case.id in targets and targets[case.id] != asdict(case):
            raise ValueError("conflicting targets within group")
        targets[case.id] = asdict(case)
        result = score_reward(sample.text, case, finish_reason=sample.finish_reason)
        if record["score"] != result["score"] or record["semantic_pass"] != result["semantic_pass"]:
            raise ValueError("saved scoring differs from recomputation")
        groups[sample.group_id].append(sample)
        results.append({"group_id": sample.group_id, "sample_index": sample.sample_index, **result})
    # Validate all records, including excluded groups; duplicate answers are retained.
    pack_rollout_groups(list(groups.values()), pad_token_id=0)
    indexed = {(row["group_id"], row["sample_index"]): row for row in results}
    summaries = []
    for group_id, samples in groups.items():
        ordered = [indexed[group_id, i] for i in range(len(samples))]
        rewards = torch.tensor([[row["reward"] for row in ordered]], dtype=torch.float64)
        advantages = group_relative_advantages(rewards)[0].tolist()
        for row, advantage in zip(ordered, advantages):
            row["advantage"] = advantage
        summaries.append({"group_id": group_id, "eligible": ordered[0]["eligible"],
                          "rewards": rewards[0].tolist(), "advantages": advantages,
                          "reward_std": rewards.std(unbiased=False).item(),
                          "zero_variance": rewards.std(unbiased=False).item() <= 1e-8})
    eligible = [group for group in summaries if group["eligible"]]
    zero = sum(group["zero_variance"] for group in eligible)
    return {"contract_version": CONTRACT_VERSION, "samples": len(results),
            "eligible_samples": sum(row["eligible"] for row in results),
            "reward_sum": sum(row["reward"] for row in results),
            "semantic_only_samples": sum(row["semantic_only"] for row in results),
            "eligible_groups": len(eligible), "zero_variance_groups": zero,
            "zero_variance_rate": zero / len(eligible) if eligible else None,
            "has_reward_learning_signal": any(not group["zero_variance"] for group in eligible),
            "record_audit_passed": True, "model_loaded": False,
            "token_decoding_verified": False, "policy_log_probs_recomputed": False,
            "groups": summaries, "records": results}
