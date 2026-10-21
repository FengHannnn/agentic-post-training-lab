"""Minimal CPU DPO with a separate frozen SFT reference and summed log probs."""

import hashlib
import math

import torch

from agentic_post_training.algorithms.losses import dpo_loss, sequence_log_probs
from agentic_post_training.data import SFTExample, collate_sft, encode_sft


def encode_pair(example, tokenizer, max_length):
    return [encode_sft(SFTExample(example.id, example.system, example.prompt,
                                 getattr(example, side)), tokenizer, max_length=max_length)
            for side in ("chosen", "rejected")]


def pair_log_probs(model, encoded, pad_token_id):
    batch = collate_sft(encoded, pad_token_id=pad_token_id)
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   use_cache=False).logits
    # Use input_ids, not HF labels (-100); shift once inside sequence_log_probs.
    return sequence_log_probs(logits, batch["input_ids"], batch["loss_mask"], average=False)


def parameter_digest(model, frozen_only=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen_only and parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(str(parameter.dtype).encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cache_reference(reference, pairs, pad_token_id):
    if any(parameter.requires_grad or parameter.grad is not None for parameter in reference.parameters()):
        raise ValueError("reference must be frozen and have no gradients")
    if not pairs:
        raise ValueError("reference pairs must be non-empty")
    reference.eval()
    result = [pair_log_probs(reference, pair, pad_token_id).detach().clone() for pair in pairs]
    if any(not torch.isfinite(value).all().item() for value in result):
        raise ValueError("reference log probabilities must be finite")
    return result


def objective(policy_log_probs, reference_log_probs, beta):
    return dpo_loss(policy_log_probs[:1], policy_log_probs[1:],
                    reference_log_probs[:1].detach(), reference_log_probs[1:].detach(), beta=beta)


def anchored_objective(policy_log_probs, reference_log_probs, beta, chosen_tokens, chosen_nll_weight=0.0):
    """DPO plus response-token-mean chosen NLL; zero weight preserves pure DPO."""
    if not math.isfinite(chosen_nll_weight) or chosen_nll_weight < 0:
        raise ValueError("chosen_nll_weight must be finite and non-negative")
    if chosen_tokens <= 0:
        raise ValueError("chosen_tokens must be positive")
    output = objective(policy_log_probs, reference_log_probs, beta)
    chosen_nll = -policy_log_probs[0] / chosen_tokens
    loss = output.loss if chosen_nll_weight == 0 else output.loss + chosen_nll_weight * chosen_nll
    return loss, output, chosen_nll


@torch.no_grad()
def evaluate_preferences(policy, pairs, reference_cache, pad_token_id, beta):
    if not pairs or len(pairs) != len(reference_cache):
        raise ValueError("non-empty pairs and matching reference cache required")
    policy.eval()
    rows = []
    for pair, cached in zip(pairs, reference_cache):
        log_probs = pair_log_probs(policy, pair, pad_token_id)
        output = objective(log_probs, cached, beta)
        rows.append({"loss": output.loss.item(), "preference_accuracy": output.preference_accuracy.item(),
                     "relative_margin": output.logits.item(),
                     "policy_margin": (log_probs[0] - log_probs[1]).item(),
                     "chosen_log_prob": log_probs[0].item(), "rejected_log_prob": log_probs[1].item()})
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("evaluation metrics must be finite")
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}


def train_dpo(policy, pairs, reference_cache, pad_token_id, *, max_steps, learning_rate,
              beta=0.1, max_grad_norm=1.0, seed=42, on_step=None, chosen_nll_weight=0.0):
    if not pairs or len(pairs) != len(reference_cache) or max_steps <= 0:
        raise ValueError("non-empty pairs, matching cache and positive steps required")
    if any(not math.isfinite(value) or value <= 0 for value in (learning_rate, beta, max_grad_norm)):
        raise ValueError("learning_rate, beta and max_grad_norm must be finite and positive")
    if not math.isfinite(chosen_nll_weight) or chosen_nll_weight < 0:
        raise ValueError("chosen_nll_weight must be finite and non-negative")
    parameters = [parameter for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not parameters or any("lora_" not in name for name, parameter in policy.named_parameters()
                             if parameter.requires_grad):
        raise ValueError("policy must train only LoRA parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    order = []
    history = []
    policy.train()
    for step in range(max_steps):
        if step % len(pairs) == 0:
            order = torch.randperm(len(pairs), generator=generator).tolist()
        index = order[step % len(pairs)]
        optimizer.zero_grad(set_to_none=True)
        log_probs = pair_log_probs(policy, pairs[index], pad_token_id)
        chosen_tokens = sum(pairs[index][0]["loss_mask"][1:])
        loss, output, chosen_nll = anchored_objective(log_probs, reference_cache[index], beta,
                                                     chosen_tokens, chosen_nll_weight)
        if not torch.isfinite(loss).item():
            raise ValueError("DPO training loss must be finite")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        record = {"step": step + 1, "pair_index": index, "loss": loss.item(),
                  "dpo_loss": output.loss.item(), "chosen_nll": chosen_nll.item(),
                  "chosen_nll_weight": chosen_nll_weight,
                  "relative_margin": output.logits.item(), "preference_accuracy": output.preference_accuracy.item(),
                  "chosen_log_prob": log_probs[0].item(), "rejected_log_prob": log_probs[1].item(),
                  "grad_norm": norm.item(), "learning_rate": learning_rate}
        history.append(record)
        if on_step:
            on_step(record)
    return history
