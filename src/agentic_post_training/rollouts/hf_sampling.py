"""CPU stochastic sampling with raw-policy and actual-behavior log probabilities."""

import math

import torch

from agentic_post_training.algorithms.losses import causal_token_log_probs
from agentic_post_training.rollouts.records import RolloutSample


@torch.no_grad()
def replay_policy_log_probs(model, prompt_ids, response_ids):
    """Score the emitted IDs directly; never decode and retokenize a response."""
    model.eval()
    ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    return causal_token_log_probs(logits, ids)[0, len(prompt_ids)-1:].detach()


@torch.no_grad()
def sample_response(model, tokenizer, case, *, group_id, sample_index, seed,
                    max_new_tokens=64, temperature=1.0, top_p=1.0):
    from transformers import GenerationConfig
    if (max_new_tokens <= 0 or not math.isfinite(temperature) or temperature <= 0
        or not math.isfinite(top_p) or not 0 < top_p <= 1):
        raise ValueError("positive length/temperature and top_p in (0,1] required")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("this initial sampler is explicitly CPU-only")
    prompt_ids = tokenizer.apply_chat_template([
        {"role": "system", "content": case.system}, {"role": "user", "content": case.prompt},
    ], tokenize=True, add_generation_prompt=True)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    config = GenerationConfig(do_sample=True, num_beams=1, num_return_sequences=1,
        temperature=temperature, top_p=top_p, top_k=0, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, use_cache=True, return_dict_in_generate=True,
        output_scores=True, output_logits=True, bos_token_id=tokenizer.raw.bos_token_id,
        eos_token_id=tokenizer.raw.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    model.eval()
    # Independent sample seeds; do not consume the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            generation_config=config, use_model_defaults=False)
    response = output.sequences[0, len(prompt_ids):].tolist()
    if not response or len(output.logits) != len(response) or len(output.scores) != len(response):
        raise ValueError("generation scores and emitted tokens must align")
    policy = [torch.log_softmax(values[0].float(), dim=-1)[token].item() for token, values in zip(response, output.logits)]
    behavior = [torch.log_softmax(values[0].float(), dim=-1)[token].item() for token, values in zip(response, output.scores)]
    result = RolloutSample(group_id, case.id, sample_index, seed, prompt_ids, response, policy, behavior,
        tokenizer.raw.eos_token_id, "eos" if response[-1] == tokenizer.raw.eos_token_id else "length",
        tokenizer.raw.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False), tokenizer.decode(response))
    replayed = replay_policy_log_probs(model, prompt_ids, response)
    original = torch.tensor(policy)
    if not torch.allclose(original, replayed, atol=1e-4, rtol=1e-4):
        raise ValueError("generated policy scores differ from teacher-forced replay")
    return result, (original-replayed).abs().max().item()
