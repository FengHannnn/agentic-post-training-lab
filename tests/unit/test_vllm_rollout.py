import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rollouts.vllm_records import sample_from_output, validate_sampling
from agentic_post_training.rewards.audit import audit_records
from agentic_post_training.rewards.contract import score_reward

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/rollout_vllm_smoke.json").read_text())
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("collect_vllm_test", ROOT / "scripts/collect_vllm_rollout.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class Tokenizer:
    eos_token_id = 0

    def decode(self, ids, **kwargs):
        return "answer" if kwargs["skip_special_tokens"] else "answer<EOS>"


def engine_output(tokens=None, finish="stop"):
    tokens = [5, 0] if tokens is None else tokens
    completion = SimpleNamespace(token_ids=tokens, finish_reason=finish,
        logprobs=[{token: SimpleNamespace(logprob=-0.1)} for token in tokens])
    return SimpleNamespace(prompt_token_ids=[1], outputs=[completion])


def adapt(output):
    case = EvalCase("hello", "greeting", "Hello", "final")
    return sample_from_output(output, [1], case, 0, 42, Tokenizer(), CONFIG)


def test_native_output_keeps_eos_and_policy_behavior_identity():
    row = adapt(engine_output())
    assert row.response_token_ids == [5, 0] and row.finish_reason == "eos"
    assert row.policy_token_log_probs == row.behavior_token_log_probs
    assert row.policy_token_log_probs is not row.behavior_token_log_probs
    truncated = adapt(engine_output([5] * 64, "length"))
    assert truncated.finish_reason == "length"
    packed = pack_rollout_groups([[row, RolloutSample(**{**asdict(truncated), "sample_index": 1})]], 0)
    assert packed["response_mask"][0, 0, 1]  # EOS is valid even when EOS==PAD.
    assert not packed["response_mask"][0, 0, 2]


@pytest.mark.parametrize("field,value", [("temperature", .8), ("top_p", .9), ("top_k", 20),
    ("group_size", 8), ("max_new_tokens", 128), ("absolute_logp_tolerance", .01)])
def test_reject_distribution_or_scope_change(field, value):
    with pytest.raises(ValueError):
        validate_sampling({**CONFIG, field: value})


@pytest.mark.parametrize("mutation", ["prompt", "outputs", "empty", "score_count", "missing_token",
    "positive_logp", "nan", "early_eos", "unexpected_stop", "short_length"])
def test_reject_malformed_engine_records(mutation):
    result = engine_output()
    c = result.outputs[0]
    if mutation == "prompt": result.prompt_token_ids = [2]
    elif mutation == "outputs": result.outputs.append(c)
    elif mutation == "empty": c.token_ids = []
    elif mutation == "score_count": c.logprobs.pop()
    elif mutation == "missing_token": c.logprobs[0] = {}
    elif mutation == "positive_logp": c.logprobs[0][5].logprob = 1
    elif mutation == "nan": c.logprobs[0][5].logprob = float("nan")
    elif mutation == "early_eos": result = engine_output([0, 5, 0])
    elif mutation == "unexpected_stop": result = engine_output([5])
    else: result = engine_output([5], "length")
    with pytest.raises(ValueError): adapt(result)


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    # Synthetic records test artifact logic, not GPU probabilities or engine execution.
    cases = [EvalCase(f"math-{g}", "math", f"Compute expression {g}", "tool_call", "3*(4+5)", 27)
             for g in range(3)] + [EvalCase("hello", "greeting", "Hello", "final")]
    raw = []
    for g, case in enumerate(cases):
        text = (json.dumps({"type": "final", "answer": "Hi"}) if g == 3 else
                json.dumps({"type": "tool_call", "name": "calculator", "arguments": {
                    "expression": "3*(4+5)" if g == 0 else "1"}}))
        result = score_reward(text, case, finish_reason="eos")
        for i in range(4):
            sample = RolloutSample(case.id, case.id, i, 42+g*1000+i, [g+1], [g+10, i+20, 0],
                [-.1]*3, [-.1]*3, 0, "eos", text, text+"<EOS>")
            raw.append({"sample": asdict(sample), "case": asdict(case), "score": result["score"],
                        "semantic_pass": result["semantic_pass"]})
    groups = [[RolloutSample(**row["sample"]) for row in raw[g*4:(g+1)*4]] for g in range(4)]
    cases = [EvalCase(**raw[g*4]["case"]) for g in range(4)]
    samples = [s for group in groups for s in group]
    class RecordedTokenizer:
        eos_token_id = samples[0].eos_token_id
        pad_token_id = 0
        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            sample = next(s for s in samples if s.response_token_ids == ids)
            return sample.text if skip_special_tokens else sample.raw_text
    monkeypatch.setattr(collector, "context", lambda: (CONFIG, tmp_path, tmp_path,
        RecordedTokenizer(), cases, [group[0].prompt_token_ids for group in groups]))
    source_name = "scripts/cloud_model_smoke.py"
    sources = {source_name: collector.file_hash(ROOT / source_name)}
    (tmp_path / "source/scripts").mkdir(parents=True)
    (tmp_path / "source" / source_name).write_bytes((ROOT / source_name).read_bytes())
    monkeypatch.setattr(collector, "inventory", lambda config, baseline: {"config": config, "source_sha256": sources})
    collector.write(tmp_path / "prerequisite_replay.json", {"single_sequence_probability_check_passed": True})
    collector.write(tmp_path / "run_manifest.json", {"config": CONFIG, "source_sha256": sources,
        "device": "fixture", "prerequisite_replay_sha256": collector.file_hash(tmp_path / "prerequisite_replay.json")})
    (tmp_path / "rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in raw), encoding="utf-8")
    first = samples[0]
    collector.write(tmp_path / "generation_checks.json", {"effective_engine": {
        "dtype": "torch.float16", "logprobs_mode": "raw_logprobs", "enforce_eager": True,
        "enable_prefix_caching": False, "enable_chunked_prefill": True, "max_model_len": 512,
        "max_num_seqs": 4, "max_num_batched_tokens": 512, "max_lora_rank": 8},
        "sampling_parameters": dict(n=1, temperature=1, top_p=1, top_k=-1, max_tokens=64,
            min_tokens=0, repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
            logprobs=1, stop_token_ids=[first.eos_token_id], ignore_eos=False, detokenize=False),
        "reproduction_probe": asdict(first), "probe_tokens_equal": True,
        "probe_logp_comparison": collector.compare_logps(first.policy_token_log_probs, first.policy_token_log_probs, .001),
        "engine_shutdown_called": True, "generation_call_seconds": 1.0})
    checks = {"records": [{"group_id": s.group_id, "sample_index": s.sample_index,
        "response_token_ids": s.response_token_ids, "replayed_policy_log_probs": s.policy_token_log_probs,
        "comparison": collector.compare_logps(s.policy_token_log_probs, s.policy_token_log_probs, .001)} for s in samples],
        "parameter_versions_unchanged": True, "frozen_no_grad": True, "forward_and_extraction_seconds": 1.0}
    collector.write(tmp_path / "replay_checks.json", checks)
    packed = pack_rollout_groups(groups, 0)
    torch.save(packed, tmp_path / "response_batch.pt")
    rewards = audit_records(raw)
    collector.write(tmp_path / "reward_audit.json", rewards)
    summary = collector.summarize(tmp_path, CONFIG, raw, groups, packed, rewards)
    collector.write(tmp_path / "summary.json", summary)
    return tmp_path, raw, groups, packed, rewards


def test_model_free_full_artifact_roundtrip(saved_run, capsys):
    directory, *_ = saved_run
    collector.audit(directory)
    assert '"artifact_audit_passed": true' in capsys.readouterr().out
    summary = collector.read(directory / "summary.json")
    assert summary["rollout_interface_checks_passed"]
    assert summary["zero_variance_rate"] == 1 and not summary["has_reward_learning_signal"]
    assert not summary["training_admitted"]


@pytest.mark.parametrize("mutation", ["duplicate_index", "seed", "text", "behavior", "reward",
    "mask", "replay_mapping", "sampling", "summary", "prerequisite", "source"])
def test_model_free_audit_rejects_corruption(saved_run, mutation):
    directory, raw, *_ = saved_run
    if mutation in ("duplicate_index", "seed", "text", "behavior"):
        if mutation == "duplicate_index": raw[1]["sample"]["sample_index"] = 0
        elif mutation == "seed": raw[1]["sample"]["seed"] += 1
        elif mutation == "text": raw[1]["sample"]["text"] = "changed"
        else: raw[1]["sample"]["behavior_token_log_probs"][0] = -1
        (directory / "rollouts.jsonl").write_text("\n".join(json.dumps(r) for r in raw), encoding="utf-8")
    elif mutation == "source":
        (directory / "source/scripts/cloud_model_smoke.py").write_text("changed", encoding="utf-8")
    elif mutation == "mask":
        packed = torch.load(directory / "response_batch.pt", weights_only=True)
        packed["response_mask"][0, 0, 0] = False
        torch.save(packed, directory / "response_batch.pt")
    else:
        filename = {"reward": "reward_audit.json", "replay_mapping": "replay_checks.json",
            "sampling": "generation_checks.json", "summary": "summary.json", "prerequisite": "prerequisite_replay.json"}[mutation]
        data = collector.read(directory / filename)
        if mutation == "reward": data["records"][0]["reward"] = 0
        elif mutation == "replay_mapping": data["records"][0]["sample_index"] = 1
        elif mutation == "sampling": data["sampling_parameters"]["min_tokens"] = 1
        elif mutation == "summary": data["samples"] = 15
        else: data["single_sequence_probability_check_passed"] = False
        (directory / filename).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError): collector.audit(directory)


def test_numeric_failure_is_preserved_not_confused_with_artifact_failure(saved_run, capsys):
    directory, raw, groups, packed, rewards = saved_run
    checks = collector.read(directory / "replay_checks.json")
    row = checks["records"][0]
    row["replayed_policy_log_probs"][0] -= .02
    row["comparison"] = collector.compare_logps(groups[0][0].policy_token_log_probs,
        row["replayed_policy_log_probs"], .001)
    (directory / "replay_checks.json").write_text(json.dumps(checks), encoding="utf-8")
    summary = collector.summarize(directory, CONFIG, raw, groups, packed, rewards)
    assert summary["replay_failed_samples"] == 1 and not summary["rollout_interface_checks_passed"]
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    collector.audit(directory)
    assert '"artifact_audit_passed": true' in capsys.readouterr().out
