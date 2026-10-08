"""Single-turn, response-only SFT preprocessing."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]: ...


@dataclass(frozen=True)
class SFTExample:
    id: str
    system: str
    prompt: str
    response: str

    def __post_init__(self) -> None:
        for name in ("id", "system", "prompt", "response"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


def load_examples(path: str | Path) -> list[SFTExample]:
    examples = []
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                example = SFTExample(**json.loads(line))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if example.id in ids:
                raise ValueError(f"duplicate example id: {example.id}")
            ids.add(example.id)
            examples.append(example)
    if not examples:
        raise ValueError("dataset must contain at least one example")
    return examples


def encode_sft(
    example: SFTExample, tokenizer: ChatTokenizer, *, max_length: int = 512
) -> dict[str, list[int]]:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    prompt_messages = [
        {"role": "system", "content": example.system},
        {"role": "user", "content": example.prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": example.response}],
        tokenize=True,
        add_generation_prompt=False,
    )
    # Never guess the boundary when a model's template is incompatible.
    if not prompt_ids or input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not provide a matching prompt prefix")
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("template must produce at least one response token")
    # Reject rather than silently cutting off a JSON tool call or its terminator.
    if len(input_ids) > max_length:
        raise ValueError(f"example {example.id} exceeds max_length={max_length}")
    loss_mask = [0] * len(prompt_ids) + [1] * (len(input_ids) - len(prompt_ids))
    labels = [token if selected else -100 for token, selected in zip(input_ids, loss_mask)]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_mask": loss_mask,
    }


def collate_sft(
    examples: list[dict[str, list[int]]], *, pad_token_id: int
) -> dict[str, Tensor]:
    """Right-pad examples; never infer padding from token IDs."""
    if not examples:
        raise ValueError("batch must not be empty")
    width = max(len(example["input_ids"]) for example in examples)
    padding_values = {
        "input_ids": pad_token_id, "attention_mask": 0, "labels": -100, "loss_mask": 0
    }
    batch = {}
    for name, padding_value in padding_values.items():
        rows = []
        for example in examples:
            length = len(example["input_ids"])
            if length == 0 or any(len(example[key]) != length for key in padding_values):
                raise ValueError("encoded fields must have the same non-zero length")
            rows.append(example[name] + [padding_value] * (width - length))
        batch[name] = torch.tensor(rows, dtype=torch.long)
    return batch
