import json

import pytest
import torch
import torch.nn.functional as F

from agentic_post_training.algorithms import masked_cross_entropy
from agentic_post_training.data import SFTExample, collate_sft, encode_sft, load_examples
from agentic_post_training.data.demo_tokenizer import DemoChatTokenizer


@pytest.fixture
def tokenizer():
    return DemoChatTokenizer()


@pytest.fixture
def example():
    return SFTExample("sample", "只输出 JSON。", "你好", '{"answer":"你好"}')


def test_only_response_and_terminator_are_trainable(example, tokenizer):
    encoded = encode_sft(example, tokenizer)
    boundary = encoded["loss_mask"].index(1)
    assert tokenizer.decode(encoded["input_ids"][boundary:]) == example.response + "\n</assistant>\n"
    assert encoded["labels"][:boundary] == [-100] * boundary
    assert encoded["labels"][boundary:] == encoded["input_ids"][boundary:]


def test_padding_does_not_mask_real_tokens(example, tokenizer):
    short = encode_sft(example, tokenizer)
    long = encode_sft(SFTExample("long", example.system, "更长的问题" * 4, "好的"), tokenizer)
    # Deliberately reuse a real token as padding, as with EOS-as-PAD models.
    batch = collate_sft([short, long], pad_token_id=short["input_ids"][-1])
    length = len(short["input_ids"])
    assert batch["attention_mask"][0, length - 1].item() == 1
    assert batch["labels"][0, length - 1].item() != -100
    assert torch.all(batch["labels"][0, length:] == -100)
    assert torch.all(batch["loss_mask"][0, length:] == 0)
    assert torch.all(batch["attention_mask"][0, length:] == 0)


def test_hf_labels_and_phase1_mask_give_same_loss(example, tokenizer):
    item = encode_sft(example, tokenizer)
    # Compact IDs make a small, deterministic vocabulary for this check.
    item["input_ids"] = [token % 7 for token in item["input_ids"]]
    item["labels"] = [token if mask else -100 for token, mask in zip(item["input_ids"], item["loss_mask"])]
    batch = collate_sft([item], pad_token_id=0)
    logits = torch.arange(batch["input_ids"].numel() * 7, dtype=torch.float32).reshape(1, -1, 7) / 100
    hf_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 7), batch["labels"][:, 1:].reshape(-1))
    manual_loss = masked_cross_entropy(logits, batch["input_ids"], batch["loss_mask"])
    assert torch.allclose(hf_loss, manual_loss)


def test_overlength_is_rejected_instead_of_truncated(example, tokenizer):
    with pytest.raises(ValueError, match="exceeds max_length"):
        encode_sft(example, tokenizer, max_length=10)


def test_incompatible_template_is_rejected(example):
    class BrokenTokenizer:
        def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
            return [1, 2] if add_generation_prompt else [9, 2, 3]

    with pytest.raises(ValueError, match="matching prompt prefix"):
        encode_sft(example, BrokenTokenizer())


@pytest.mark.parametrize("field", ["id", "system", "prompt", "response"])
def test_blank_fields_are_rejected(example, field):
    values = vars(example).copy()
    values[field] = " "
    with pytest.raises(ValueError, match=field):
        SFTExample(**values)


def test_duplicate_ids_are_rejected(tmp_path, example):
    path = tmp_path / "samples.jsonl"
    row = json.dumps(vars(example), ensure_ascii=False)
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_examples(path)


def test_bad_json_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("\ninvalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        load_examples(path)


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        collate_sft([], pad_token_id=0)
