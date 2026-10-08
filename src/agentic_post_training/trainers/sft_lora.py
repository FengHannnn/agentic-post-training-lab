"""Response-only SFT helpers; HF/PEFT imports remain optional."""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from agentic_post_training.data import SFTExample, collate_sft


def check_split(train: list[SFTExample], evaluation: list[SFTExample]) -> None:
    if {item.id for item in train} & {item.id for item in evaluation}:
        raise ValueError("train and eval contain overlapping IDs")
    if {(item.system.strip(), item.prompt.strip()) for item in train} & {
        (item.system.strip(), item.prompt.strip()) for item in evaluation
    }:
        raise ValueError("train and eval contain overlapping prompts")


def attach_lora(
    model: nn.Module, *, rank: int = 4, alpha: int = 8,
    target_modules: list[str] | None = None,
) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
        lora_dropout=0.0, target_modules=target_modules or ["q_proj", "v_proj"],
        bias="none",
    ))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("this experiment must train only LoRA parameters")
    return model


def model_inputs(batch: dict[str, Tensor]) -> dict[str, Tensor]:
    # loss_mask is for Phase 1, not a keyword argument of the HF model.
    return {name: batch[name] for name in ("input_ids", "attention_mask", "labels")}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module, examples: list[dict[str, list[int]]], *, pad_token_id: int,
) -> float:
    if not examples:
        raise ValueError("evaluation data must not be empty")
    model.eval()
    weighted_loss = 0.0
    token_count = 0
    for example in examples:
        batch = collate_sft([example], pad_token_id=pad_token_id)
        count = int((batch["labels"][:, 1:] != -100).sum().item())
        loss = model(**model_inputs(batch)).loss.item()
        if count == 0 or not math.isfinite(loss):
            raise ValueError("evaluation requires response tokens and finite loss")
        weighted_loss += loss * count
        token_count += count
    return weighted_loss / token_count


def train_steps(
    model: nn.Module, examples: list[dict[str, list[int]]], *, pad_token_id: int,
    max_steps: int = 3, batch_size: int = 1, learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    on_step: Callable[[dict[str, float | int]], None] | None = None,
) -> list[dict[str, float | int]]:
    if not examples or max_steps <= 0 or batch_size <= 0:
        raise ValueError("non-empty data and positive steps/batch_size required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    model.train()
    history = []
    for step in range(max_steps):
        selected = [examples[(step * batch_size + index) % len(examples)] for index in range(batch_size)]
        batch = collate_sft(selected, pad_token_id=pad_token_id)
        optimizer.zero_grad(set_to_none=True)
        # HF causal models handle the label shift internally.
        loss = model(**model_inputs(batch)).loss
        if not torch.isfinite(loss).item():
            raise ValueError("training loss is not finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        record = {
            "step": step + 1, "loss": loss.detach().item(),
            "grad_norm": gradient_norm.item(), "learning_rate": learning_rate,
            "response_tokens": int((batch["labels"][:, 1:] != -100).sum().item()),
        }
        history.append(record)
        if on_step is not None:
            on_step(record)
    return history
