"""Load real chat tokenizers without loading model weights."""

from pathlib import Path
from typing import Any


DEFAULT_TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"


class HFChatTokenizer:
    """Keep the list-of-IDs contract explicit across library versions."""

    def __init__(self, tokenizer: Any) -> None:
        self.raw = tokenizer
        if not tokenizer.chat_template:
            raise ValueError("tokenizer must provide a chat template")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must provide a PAD or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    @property
    def pad_token_id(self) -> int:
        return self.raw.pad_token_id

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize:
            raise ValueError("this adapter returns token IDs only")
        return self.raw.apply_chat_template(
            conversation, tokenize=True,
            add_generation_prompt=add_generation_prompt, return_dict=False,
        )

    def decode(self, token_ids: list[int]) -> str:
        return self.raw.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


def load_hf_tokenizer(
    model: str = DEFAULT_TOKENIZER, *, cache_dir: str | Path | None = None,
    revision: str = "main", local_files_only: bool = False,
) -> HFChatTokenizer:
    # Lazy import keeps Phase 1 and offline teaching tests independent of HF.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model, cache_dir=str(cache_dir) if cache_dir is not None else None,
        revision=revision, local_files_only=local_files_only,
        trust_remote_code=False,
    )
    return HFChatTokenizer(tokenizer)
