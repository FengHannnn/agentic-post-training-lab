"""Separate sampler behavior scores from same-path trainer old/new scores."""

import torch

from agentic_post_training.algorithms.losses import causal_token_log_probs
from agentic_post_training.rollouts.records import pack_rollout_groups


def score_response(model, prompt_ids, response_ids, *, grad_enabled):
    if not prompt_ids or not response_ids or model.training:
        raise ValueError("non-empty sequence and eval-mode model required")
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=next(model.parameters()).device)
    with torch.set_grad_enabled(grad_enabled):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        return causal_token_log_probs(logits.float(), ids)[0, len(prompt_ids) - 1:]


def no_update_metrics(old, new, tolerance):
    if old.ndim != 1 or old.numel() == 0 or new.shape != old.shape:
        raise ValueError("aligned non-empty response scores required")
    if old.requires_grad or not new.requires_grad:
        raise ValueError("old must be detached and new must retain autograd")
    if old.dtype != torch.float32 or new.dtype != torch.float32:
        raise ValueError("this trainer score contract requires float32")
    if not 0 < tolerance < 1:
        raise ValueError("invalid logp tolerance")
    if not torch.isfinite(old).all() or not torch.isfinite(new).all() or (old > 1e-6).any() or (new > 1e-6).any():
        raise ValueError("invalid policy log probabilities")
    # The ratio used by a future policy objective retains the new-policy graph.
    live_ratio = (new - old.detach()).exp()
    delta = new.detach().double() - old.detach().double()
    ratios = delta.exp()
    return {"max_abs_logp_delta": delta.abs().max().item(),
            "max_abs_ratio_minus_one": (ratios - 1).abs().max().item(),
            "ratio_min": ratios.min().item(), "ratio_max": ratios.max().item(),
            "passed": delta.abs().max().item() <= tolerance,
            "old_requires_grad": old.requires_grad, "new_requires_grad": new.requires_grad,
            "ratio_requires_grad": live_ratio.requires_grad,
            "token_ratios": ratios.cpu().tolist()}


def sampler_trainer_difference(behavior, trainer_old):
    q = torch.as_tensor(behavior, dtype=torch.float64)
    p = torch.as_tensor(trainer_old, dtype=torch.float64)
    if q.ndim != 1 or p.shape != q.shape or not q.numel() or not torch.isfinite(q).all() or not torch.isfinite(p).all():
        raise ValueError("finite aligned sampler/trainer scores required")
    delta = p - q
    if (q > 1e-6).any() or (p > 1e-6).any():
        raise ValueError("log probabilities must not be positive")
    return {"max_abs_logp_delta": delta.abs().max().item(),
            "per_token_log_p_over_q": delta.tolist(), "sequence_log_p_over_q": delta.sum().item(),
            "behavior_correction_applied": False}


def pack_trainer_scores(groups, records, pad_token_id):
    original = pack_rollout_groups(groups, pad_token_id)
    keys = {(sample.group_id, sample.sample_index) for group in groups for sample in group}
    indexed = {(row["group_id"], row["sample_index"]): row for row in records}
    if len(indexed) != len(records) or set(indexed) != keys:
        raise ValueError("trainer scores must cover each original sample exactly once")
    mask = original["response_mask"]
    old, new, ratios = (torch.zeros(mask.shape, dtype=torch.float32) for _ in range(3))
    eligible_mask = torch.zeros_like(mask)
    for g, group in enumerate(groups):
        for sample in group:
            row = indexed[sample.group_id, sample.sample_index]
            n, i = len(sample.response_token_ids), sample.sample_index
            if (row["response_token_ids"] != sample.response_token_ids
                or len(row["trainer_old_token_log_probs"]) != n or len(row["trainer_new_token_log_probs"]) != n):
                raise ValueError("trainer score IDs or lengths differ from sampler")
            old[g, i, :n] = torch.tensor(row["trainer_old_token_log_probs"])
            new[g, i, :n] = torch.tensor(row["trainer_new_token_log_probs"])
            if (not torch.isfinite(old[g, i, :n]).all() or not torch.isfinite(new[g, i, :n]).all()
                or (old[g, i, :n] > 1e-6).any() or (new[g, i, :n] > 1e-6).any()):
                raise ValueError("invalid trainer scores")
            ratios[g, i, :n] = (new[g, i, :n] - old[g, i, :n]).exp()
            eligible_mask[g, i, :n] = row["eligible"]
    return {"response_token_ids": original["response_token_ids"], "response_mask": mask,
            "eligible_response_mask": eligible_mask,
            "sampler_behavior_log_probs": original["behavior_token_log_probs"],
            "trainer_old_log_probs": old, "trainer_new_log_probs": new, "policy_ratios": ratios}
