"""An offline teaching tokenizer, not a tokenizer for any pretrained model."""


class DemoChatTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize:
            raise ValueError("demo supports token IDs only")
        text = "".join(
            f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n"
            for message in conversation
        )
        if add_generation_prompt:
            text += "<assistant>\n"
        return [ord(character) + 1 for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token - 1) for token in token_ids if token != self.pad_token_id)
