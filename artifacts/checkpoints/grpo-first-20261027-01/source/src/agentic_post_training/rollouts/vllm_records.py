"""Adapt native vLLM outputs under one explicit unmodified sampling contract."""

from agentic_post_training.rollouts.records import RolloutSample


def validate_sampling(config):
    # Only this configuration permits behavior logp == raw policy logp here.
    if (config["temperature"] != 1 or config["top_p"] != 1 or config["top_k"] != -1
        or config["group_size"] != 4 or config["max_new_tokens"] != 64
        or config["base_seed"] != 42 or config["seed_group_stride"] != 1000
        or config["dtype"] != "float16" or config["absolute_logp_tolerance"] != 0.001
        or config["max_model_len"] != 512 or config["max_num_seqs"] != 4
        or config["max_num_batched_tokens"] != 512 or config["gpu_memory_utilization"] != 0.25):
        raise ValueError("configuration differs from the fixed 4x4 unmodified sampling contract")


def sample_from_output(output, prompt, case, index, seed, tokenizer, config):
    validate_sampling(config)
    if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
        raise ValueError("engine changed prompt IDs or output count")
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    if not tokens or len(tokens) > config["max_new_tokens"]:
        raise ValueError("invalid response length")
    if completion.logprobs is None or len(completion.logprobs) != len(tokens):
        raise ValueError("missing or misaligned engine scores")
    if any(token not in values for token, values in zip(tokens, completion.logprobs)):
        raise ValueError("emitted token missing from logprobs")
    logps = [values[token].logprob for token, values in zip(tokens, completion.logprobs)]
    eos = tokenizer.eos_token_id
    if completion.finish_reason == "stop" and tokens[-1] == eos:
        finish = "eos"
    elif completion.finish_reason == "length" and len(tokens) == config["max_new_tokens"]:
        finish = "length"
    else:
        raise ValueError("unsupported stop or inconsistent truncation")
    return RolloutSample(case.id, case.id, index, seed, list(prompt), tokens, logps, list(logps),
        eos, finish,
        tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False))
