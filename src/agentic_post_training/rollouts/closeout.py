"""Frozen Phase 4 closeout selection and native-record validation."""

from dataclasses import asdict
import math

from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rewards.audit import audit_records


def select_cases(train, identifiers):
    indexed = {row.id: row for row in train}
    if len(identifiers) != 48 or len(set(identifiers)) != 48:
        raise ValueError("exactly 48 unique pinned training inputs required")
    if not set(identifiers) <= set(indexed):
        raise ValueError("closeout inputs must belong to training split")
    cases = [indexed[key].target() for key in identifiers]
    if any(case.expected_type != "tool_call" for case in cases):
        raise ValueError("closeout signal diagnostic is arithmetic only")
    return cases


def sample_seed(group, index):
    return 42 + group * 1000 + index


def budget_limits(price, spent):
    if not math.isfinite(price) or price<=0 or not math.isfinite(spent) or not 0<=spent<200:
        raise ValueError("positive price and cumulative spent below 200 required")
    budget=min(10.,200-spent)
    seconds=min(1800.,budget/price*3600)-10  # Reserve cleanup time.
    if seconds<=0:
        raise ValueError("remaining budget is too small for bounded startup/cleanup")
    return budget,seconds


def validate_records(rows, cases, prompts, tokenizer, config):
    if len(rows) != len(cases) * 4 or len(prompts) != len(cases):
        raise ValueError("incomplete rollout groups")
    groups = []
    for g, case in enumerate(cases):
        group = []
        for i, row in enumerate(rows[g*4:(g+1)*4]):
            sample = RolloutSample(**row["sample"])
            if (row["case"] != asdict(case) or sample.group_id != case.id or sample.case_id != case.id
                or sample.sample_index != i or sample.seed != sample_seed(g, i)
                or sample.prompt_token_ids != prompts[g] or sample.eos_token_id != tokenizer.eos_token_id
                or sample.policy_token_log_probs != sample.behavior_token_log_probs
                or len(sample.response_token_ids) > config["max_new_tokens"]
                or (sample.finish_reason == "length" and len(sample.response_token_ids) != config["max_new_tokens"])
                or sample.text != tokenizer.decode(sample.response_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                or sample.raw_text != tokenizer.decode(sample.response_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)):
                raise ValueError("native token, identity, sampling or decoding mismatch")
            group.append(sample)
        groups.append(group)
    return groups, pack_rollout_groups(groups, tokenizer.pad_token_id), audit_records(rows)


def benchmark_summary(batch_samples, serial_samples, batch_seconds, serial_seconds, tolerance=.001):
    if (not batch_samples or len(batch_samples) != len(serial_samples)
        or batch_seconds <= 0 or serial_seconds <= 0):
        raise ValueError("paired samples and positive timing required")
    comparisons = []
    for batch, serial in zip(batch_samples, serial_samples):
        if (batch.prompt_token_ids != serial.prompt_token_ids or batch.seed != serial.seed
            or batch.case_id != serial.case_id or batch.sample_index != serial.sample_index):
            raise ValueError("benchmark input/seed mapping differs")
        equal = batch.response_token_ids == serial.response_token_ids
        error = max(abs(x-y) for x,y in zip(batch.policy_token_log_probs, serial.policy_token_log_probs)) if equal else None
        comparisons.append({"tokens_equal": equal, "max_logp_error": error,
                            "within_original_tolerance": equal and error <= tolerance})
    return {"paired_requests": len(batch_samples), "batch_seconds": batch_seconds,
        "serial_seconds": serial_seconds, "batch_over_serial_speedup": serial_seconds/batch_seconds,
        "batch_response_tokens_per_second": sum(len(s.response_token_ids) for s in batch_samples)/batch_seconds,
        "serial_response_tokens_per_second": sum(len(s.response_token_ids) for s in serial_samples)/serial_seconds,
        "token_reproduction_rate": sum(c["tokens_equal"] for c in comparisons)/len(comparisons),
        "logp_reproduction_rate": sum(c["within_original_tolerance"] for c in comparisons)/len(comparisons),
        "comparisons": comparisons}
