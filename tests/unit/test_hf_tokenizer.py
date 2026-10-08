import pytest

from agentic_post_training.data.hf_tokenizer import HFChatTokenizer


class StubTokenizer:
    chat_template = "template"
    pad_token_id = None
    eos_token_id = 2
    eos_token = "<eos>"

    @property
    def pad_token(self):
        return getattr(self, "_pad_token", None)

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value
        self.pad_token_id = self.eos_token_id

    def apply_chat_template(self, conversation, **kwargs):
        self.options = kwargs
        return [1, 2]


def test_missing_pad_reuses_eos_and_preserves_list_contract():
    raw = StubTokenizer()
    tokenizer = HFChatTokenizer(raw)
    assert tokenizer.pad_token_id == 2
    assert raw.padding_side == "right"
    assert tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True) == [1, 2]
    assert raw.options["return_dict"] is False


def test_existing_pad_is_preserved():
    raw = StubTokenizer()
    raw.pad_token_id = 7
    assert HFChatTokenizer(raw).pad_token_id == 7


def test_missing_template_is_rejected():
    raw = StubTokenizer()
    raw.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        HFChatTokenizer(raw)


def test_missing_pad_and_eos_is_rejected():
    raw = StubTokenizer()
    raw.eos_token_id = None
    with pytest.raises(ValueError, match="PAD or EOS"):
        HFChatTokenizer(raw)
