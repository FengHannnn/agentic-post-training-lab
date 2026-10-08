"""Shared deterministic CPU generation for evaluation scripts."""

import torch


@torch.no_grad()
def generate_prediction(model, tokenizer, case, max_new_tokens):
    ids = tokenizer.apply_chat_template([
        {"role": "system", "content": case.system},
        {"role": "user", "content": case.prompt},
    ], tokenize=True, add_generation_prompt=True)
    inputs = torch.tensor([ids], dtype=torch.long)
    model.eval()
    output = model.generate(
        input_ids=inputs, attention_mask=torch.ones_like(inputs),
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.raw.eos_token_id,
    )
    generated = output[0, len(ids):].tolist()
    return {
        "prediction": tokenizer.raw.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        "raw_prediction": tokenizer.decode(generated),
        "generated_tokens": len(generated),
        "hit_token_limit": len(generated) >= max_new_tokens and generated[-1] != tokenizer.raw.eos_token_id,
    }
