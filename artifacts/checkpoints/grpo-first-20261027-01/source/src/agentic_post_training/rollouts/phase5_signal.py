"""Phase 5 training-only 144x8 signal diagnostic; no policy updates."""

from dataclasses import asdict
import math

from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rewards.audit import audit_records


def select_cases(train, identifiers):
    indexed = {row.id: row for row in train if row.expected_type == "tool_call"}
    if len(indexed) != 144 or identifiers != sorted(indexed):
        raise ValueError("exactly all 144 arithmetic training inputs in sorted ID order required")
    return [indexed[key].target() for key in identifiers]


def sample_seed(group, index):
    return 20261021 + group * 1000 + index


def sample_from_output(output, prompt, case, index, seed, tokenizer):
    if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
        raise ValueError("native prompt IDs or completion count differs")
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    if not tokens or len(tokens) > 64 or completion.logprobs is None or len(completion.logprobs) != len(tokens):
        raise ValueError("missing or misaligned native tokens/scores")
    if any(token not in values for token, values in zip(tokens, completion.logprobs)):
        raise ValueError("emitted token absent from native scores")
    logps = [values[token].logprob for token, values in zip(tokens, completion.logprobs)]
    if completion.finish_reason == "stop" and tokens[-1] == tokenizer.eos_token_id:
        finish = "eos"
    elif completion.finish_reason == "length" and len(tokens) == 64:
        finish = "length"
    else:
        raise ValueError("unsupported stop condition")
    return RolloutSample(case.id,case.id,index,seed,list(prompt),tokens,logps,list(logps),
        tokenizer.eos_token_id,finish,
        tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False),
        tokenizer.decode(tokens,skip_special_tokens=False,clean_up_tokenization_spaces=False))


def validate_records(rows, cases, prompts, tokenizer):
    if len(rows) != len(cases)*8 or len(prompts) != len(cases):
        raise ValueError("all eight samples per input must be retained")
    groups = []
    for g,case in enumerate(cases):
        group = []
        for i,row in enumerate(rows[g*8:(g+1)*8]):
            s = RolloutSample(**row["sample"])
            if (row["case"] != asdict(case) or s.group_id != case.id or s.case_id != case.id
                or s.sample_index != i or s.seed != sample_seed(g,i) or s.prompt_token_ids != prompts[g]
                or s.policy_token_log_probs != s.behavior_token_log_probs
                or s.eos_token_id != tokenizer.eos_token_id or len(s.response_token_ids)>64
                or (s.finish_reason == "length" and len(s.response_token_ids)!=64)
                or s.text != tokenizer.decode(s.response_token_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                or s.raw_text != tokenizer.decode(s.response_token_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)):
                raise ValueError("sample identity, native tokens, decode or sampling contract differs")
            group.append(s)
        groups.append(group)
    return groups,pack_rollout_groups(groups,tokenizer.pad_token_id),audit_records(rows)


def signal_summary(rewards):
    counts={"all_correct":0,"all_wrong":0,"mixed":0}
    prefix_mixed = 0
    by_style = {}
    for group in rewards["groups"]:
        values = group["rewards"]
        if len(values)!=8 or any(v not in (0.,1.) for v in values):
            raise ValueError("eight binary arithmetic rewards required")
        kind="all_correct" if min(values)==1 else "all_wrong" if max(values)==0 else "mixed"
        counts[kind]+=1
        prefix_mixed+=len(set(values[:4]))>1
        style=group["group_id"].split('-')[-2]
        entry=by_style.setdefault(style,{"groups":0,"successes":0,"samples":0,"mixed":0})
        entry["groups"]+=1;entry["successes"]+=sum(values);entry["samples"]+=8;entry["mixed"]+=kind=="mixed"
    return {"group_outcomes":counts,"prefix4_mixed_groups":prefix_mixed,
        "full8_mixed_groups":counts["mixed"],"extra_mixed_groups_from_samples_5_to_8":counts["mixed"]-prefix_mixed,
        "samples_in_nonzero_advantage_groups":counts["mixed"]*8,"by_style":by_style}


def run_seconds(price):
    if not math.isfinite(price) or price<=0:
        raise ValueError("finite positive hourly price required")
    seconds=min(1800.,10./price*3600)-10
    if seconds<=0: raise ValueError("price exceeds finite startup/cleanup budget")
    return seconds
