"""Token-preserving rollout records and explicit length-based response masks."""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutSample:
    group_id: str
    case_id: str
    sample_index: int
    seed: int
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    policy_token_log_probs: list[float]
    behavior_token_log_probs: list[float]
    eos_token_id: int
    finish_reason: str
    text: str
    raw_text: str

    def __post_init__(self):
        if not self.group_id or not self.case_id or self.sample_index < 0 or self.seed < 0:
            raise ValueError("non-empty IDs and non-negative sample index/seed required")
        for tokens in (self.prompt_token_ids, self.response_token_ids):
            if not tokens or any(type(token) is not int or token < 0 for token in tokens):
                raise ValueError("non-empty valid token IDs required")
        for values in (self.policy_token_log_probs, self.behavior_token_log_probs):
            if len(values) != len(self.response_token_ids) or any(not math.isfinite(value) or value > 1e-6 for value in values):
                raise ValueError("one finite log probability per response token required")
        if self.finish_reason not in ("eos", "length"):
            raise ValueError("finish reason must be eos or length")
        if self.eos_token_id in self.response_token_ids[:-1]:
            raise ValueError("tokens after EOS are not a generated response")
        if (self.response_token_ids[-1] == self.eos_token_id) != (self.finish_reason == "eos"):
            raise ValueError("EOS and finish reason disagree")


def pack_rollout_groups(groups, pad_token_id):
    """Return [prompt, generation, response_token] tensors, retaining real EOS."""
    if not groups or len(groups[0]) < 2 or any(len(group) != len(groups[0]) for group in groups):
        raise ValueError("non-empty equal-sized groups with at least two samples required")
    if len({group[0].group_id for group in groups}) != len(groups):
        raise ValueError("group IDs must be unique")
    for group in groups:
        if (len({(row.group_id, row.case_id, tuple(row.prompt_token_ids)) for row in group}) != 1
            or {row.sample_index for row in group} != set(range(len(group)))):
            raise ValueError("group must share input and contain each sample index once")
    width = max(len(row.response_token_ids) for group in groups for row in group)
    shape = (len(groups), len(groups[0]), width)
    tokens = torch.full(shape, pad_token_id, dtype=torch.long)
    mask = torch.zeros(shape, dtype=torch.bool)
    policy = torch.zeros(shape, dtype=torch.float32)
    behavior = torch.zeros(shape, dtype=torch.float32)
    for batch_index, group in enumerate(groups):
        for row in group:
            n = len(row.response_token_ids)
            slot = (batch_index, row.sample_index, slice(0, n))
            tokens[slot] = torch.tensor(row.response_token_ids)
            mask[slot] = True
            policy[slot] = torch.tensor(row.policy_token_log_probs)
            behavior[slot] = torch.tensor(row.behavior_token_log_probs)
    return {"response_token_ids": tokens, "response_mask": mask,
        "policy_token_log_probs": policy, "behavior_token_log_probs": behavior}
