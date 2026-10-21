"""Detached token TIS weights for a clipped policy surrogate, without KL.

This conditional-token approximation does not correct prefix distributions or
the joint distribution of samples used for group-relative advantages.
"""

import math

import torch


def _scores_on_mask(scores, mask, name):
    if scores.shape != mask.shape or not scores.is_floating_point():
        raise ValueError(f"{name} must be floating scores aligned with mask")
    selected = scores[mask]
    if not torch.isfinite(selected).all() or (selected > 1e-6).any():
        raise ValueError(f"{name} must contain finite non-positive active log probabilities")
    # Replace padding/excluded values before exponentiation, including NaNs.
    return torch.where(mask, scores, torch.zeros_like(scores))


def token_tis_weights(old_log_probs, behavior_log_probs, mask, *, upper_cap=2.0):
    if (mask.dtype != torch.bool or mask.ndim != 3 or not mask.any()
        or not math.isfinite(upper_cap) or upper_cap < 1):
        raise ValueError("non-empty boolean [group, sample, token] mask and cap >= 1 required")
    if old_log_probs.requires_grad or behavior_log_probs.requires_grad:
        raise ValueError("old and behavior probabilities must be fixed")
    old = _scores_on_mask(old_log_probs, mask, "old")
    behavior = _scores_on_mask(behavior_log_probs, mask, "behavior")
    # Clamp in log space before exp so a large positive delta cannot overflow.
    log_weights = old.double() - behavior.double()
    weights = log_weights.clamp(max=math.log(upper_cap)).exp()
    weights = torch.where(mask, weights, torch.zeros_like(weights)).detach()
    active = weights[mask]
    return weights, {
        "min_weight": active.min().item(), "max_weight": active.max().item(),
        "mean_weight": active.mean().item(),
        "cap_fraction": (log_weights[mask] > math.log(upper_cap)).double().mean().item(),
        "weights_require_grad": weights.requires_grad,
    }


def corrected_grpo_policy_loss(new_log_probs, old_log_probs, behavior_log_probs,
                               advantages, mask, *, upper_cap=2.0, clip_epsilon=0.2):
    """Eligible-token mean of -w * min(r*A, clip(r)*A); no KL or update.

    w = stopgrad(min(cap, p_old / q_sampler)); r = p_new / p_old.
    old, behavior and completion-level advantages must be detached. Padding
    and excluded samples never enter exponentiation or the denominator.
    """
    if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip epsilon must lie in (0, 1)")
    weights, diagnostics = token_tis_weights(old_log_probs, behavior_log_probs, mask, upper_cap=upper_cap)
    if advantages.shape != mask.shape[:2] or advantages.requires_grad:
        raise ValueError("fixed completion-level advantages required")
    active_samples = mask.any(dim=-1)
    if not torch.isfinite(advantages[active_samples]).all():
        raise ValueError("finite active advantages required")
    old = _scores_on_mask(old_log_probs, mask, "old")
    new = _scores_on_mask(new_log_probs, mask, "new")
    adv = torch.where(active_samples, advantages, torch.zeros_like(advantages)).unsqueeze(-1)
    ratios = (new - old).exp()
    if not torch.isfinite(ratios[mask]).all():
        raise ValueError("policy ratios overflow; refusing non-finite objective")
    objective = torch.minimum(ratios * adv, ratios.clamp(1-clip_epsilon, 1+clip_epsilon) * adv)
    loss = -(weights.to(new.dtype) * objective)[mask].mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite corrected policy loss")
    return loss, diagnostics
