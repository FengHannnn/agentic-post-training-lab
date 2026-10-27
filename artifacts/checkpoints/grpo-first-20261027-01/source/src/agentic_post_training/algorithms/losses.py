"""Minimal post-training objectives implemented directly with PyTorch.

The functions in this module deliberately operate on tensors instead of model
objects. This keeps masking, log-probability alignment, and policy ratios easy
to inspect and test before a trainer framework is introduced.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_causal_inputs(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels must have shape [batch, sequence]")
    if loss_mask.shape != labels.shape:
        raise ValueError("loss_mask must have the same shape as labels")
    if logits.shape[1] < 2:
        raise ValueError("causal objectives require at least two sequence positions")


def causal_token_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """Return next-token log probabilities with shape ``[batch, sequence - 1]``.

    Logits at position ``t`` predict the label at position ``t + 1``. Keeping
    this shift inside one function prevents a common silent training bug.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError(
            "expected logits [batch, sequence, vocabulary] and labels [batch, sequence]"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal log probabilities require at least two positions")

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    return log_probs.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)


def sequence_log_probs(
    logits: Tensor,
    labels: Tensor,
    loss_mask: Tensor,
    *,
    average: bool = False,
) -> Tensor:
    """Return each sequence's masked response log probability.

    ``loss_mask[:, t]`` describes whether ``labels[:, t]`` belongs to the
    trainable response. Position zero is discarded because no preceding logit
    predicts it.
    """

    _validate_causal_inputs(logits, labels, loss_mask)
    token_log_probs = causal_token_log_probs(logits, labels)
    shifted_mask = loss_mask[:, 1:].to(dtype=token_log_probs.dtype)
    totals = (token_log_probs * shifted_mask).sum(dim=-1)

    if not average:
        return totals

    counts = shifted_mask.sum(dim=-1)
    if torch.any(counts == 0):
        raise ValueError("every sequence must contain at least one response token")
    return totals / counts


def masked_cross_entropy(logits: Tensor, labels: Tensor, loss_mask: Tensor) -> Tensor:
    """Compute mean next-token cross entropy over response tokens only."""

    _validate_causal_inputs(logits, labels, loss_mask)
    token_log_probs = causal_token_log_probs(logits, labels)
    shifted_mask = loss_mask[:, 1:].to(dtype=token_log_probs.dtype)
    token_count = shifted_mask.sum()
    if token_count.item() == 0:
        raise ValueError("loss_mask must select at least one response token")
    return -(token_log_probs * shifted_mask).sum() / token_count


@dataclass(frozen=True)
class DPOLossOutput:
    loss: Tensor
    logits: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    preference_accuracy: Tensor


def dpo_loss(
    policy_chosen_log_probs: Tensor,
    policy_rejected_log_probs: Tensor,
    reference_chosen_log_probs: Tensor,
    reference_rejected_log_probs: Tensor,
    *,
    beta: float = 0.1,
) -> DPOLossOutput:
    """Compute the standard sigmoid DPO objective for a batch of preference pairs."""

    tensors = (
        policy_chosen_log_probs,
        policy_rejected_log_probs,
        reference_chosen_log_probs,
        reference_rejected_log_probs,
    )
    expected_shape = policy_chosen_log_probs.shape
    for tensor in tensors:
        if tensor.ndim != 1:
            raise ValueError("all DPO log-probability tensors must be one-dimensional")
        if tensor.shape != expected_shape:
            raise ValueError("all DPO log-probability tensors must have the same shape")
    if beta <= 0:
        raise ValueError("beta must be positive")

    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    reference_log_ratios = (
        reference_chosen_log_probs - reference_rejected_log_probs
    )
    preference_logits = beta * (policy_log_ratios - reference_log_ratios)
    losses = -F.logsigmoid(preference_logits)

    chosen_rewards = beta * (
        policy_chosen_log_probs - reference_chosen_log_probs
    ).detach()
    rejected_rewards = beta * (
        policy_rejected_log_probs - reference_rejected_log_probs
    ).detach()

    return DPOLossOutput(
        loss=losses.mean(),
        logits=preference_logits.detach(),
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        preference_accuracy=(chosen_rewards > rejected_rewards).float().mean(),
    )


def group_relative_advantages(rewards: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize rewards within each prompt group.

    Args:
        rewards: Tensor shaped ``[batch, generations]``.
        eps: Groups with a standard deviation no larger than ``eps`` receive
            zero advantages because they contain no relative learning signal.
    """

    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [batch, generations]")
    if rewards.shape[1] < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    if eps <= 0:
        raise ValueError("eps must be positive")

    means = rewards.mean(dim=1, keepdim=True)
    standard_deviations = rewards.std(dim=1, keepdim=True, unbiased=False)
    normalized = (rewards - means) / standard_deviations.clamp_min(eps)
    return torch.where(
        standard_deviations > eps,
        normalized,
        torch.zeros_like(normalized),
    )


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: Tensor
    mean_kl: Tensor
    clip_fraction: Tensor
    ratios: Tensor


def grpo_clipped_loss(
    new_log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    reference_log_probs: Tensor | None = None,
    beta: float = 0.0,
) -> GRPOLossOutput:
    """Compute a token-level clipped GRPO policy objective.

    Inputs use shape ``[batch, generations, response_tokens]``. Completion-level
    advantages shaped ``[batch, generations]`` are broadcast across tokens.
    The optional KL term uses the non-negative estimator ``exp(x) - x - 1``,
    where ``x = reference_log_prob - policy_log_prob``.
    """

    if new_log_probs.shape != old_log_probs.shape:
        raise ValueError("new_log_probs and old_log_probs must have the same shape")
    if new_log_probs.ndim != 3:
        raise ValueError("policy log probabilities must have shape [batch, group, tokens]")
    if response_mask.shape != new_log_probs.shape:
        raise ValueError("response_mask must have the same shape as policy log probabilities")
    if advantages.shape == new_log_probs.shape[:2]:
        advantages = advantages.unsqueeze(-1)
    elif advantages.shape != new_log_probs.shape:
        raise ValueError("advantages must be completion-level or token-level")
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if reference_log_probs is not None and reference_log_probs.shape != new_log_probs.shape:
        raise ValueError("reference_log_probs must match policy log-probability shape")

    mask = response_mask.to(dtype=new_log_probs.dtype)
    token_count = mask.sum()
    if token_count.item() == 0:
        raise ValueError("response_mask must select at least one token")

    log_ratio = new_log_probs - old_log_probs
    ratios = log_ratio.exp()
    unclipped = ratios * advantages
    clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    clipped = clipped_ratios * advantages
    policy_objective = torch.minimum(unclipped, clipped)

    if reference_log_probs is None:
        per_token_kl = torch.zeros_like(new_log_probs)
    else:
        reference_ratio_log = reference_log_probs - new_log_probs
        per_token_kl = reference_ratio_log.exp() - reference_ratio_log - 1.0

    objective = policy_objective - beta * per_token_kl
    loss = -(objective * mask).sum() / token_count
    clipped_tokens = (ratios != clipped_ratios).to(mask.dtype)

    return GRPOLossOutput(
        loss=loss,
        mean_kl=(per_token_kl * mask).sum().detach() / token_count,
        clip_fraction=(clipped_tokens * mask).sum().detach() / token_count,
        ratios=ratios.detach(),
    )
