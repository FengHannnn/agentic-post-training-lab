"""Fixed 4x4 sampling, process-isolated GPU replay, and model-free audit."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verify_cloud_bundle import file_hash
from replay_cloud_smoke import compare_logps, response_logps
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rollouts.vllm_records import validate_sampling, sample_from_output
from agentic_post_training.evaluation.tool_calls import score_output
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rewards.audit import audit_records


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def context():
    from verify_cloud_bundle import main as verify_bundle
    from transformers import AutoTokenizer
    verify_bundle()
    config = read(ROOT / "configs/rollout_vllm_smoke.json")
    validate_sampling(config)
    old = read(ROOT / "configs/rollout_cpu_smoke.json")
    baseline = ROOT / old["baseline"]
    manifest = read(baseline / "manifest.json")
    folder = ROOT / old["preference_data"]
    data_manifest = read(folder / "manifest.json")
    if data_manifest["baseline_manifest_sha256"] != file_hash(baseline / "manifest.json"):
        raise ValueError("data and model identities differ")
    train = {row.id: row for row in load_preference_splits(folder, data_manifest)["train"]}
    cases = [train[identifier].target() for identifier in old["case_ids"]]
    if len(cases) != 4 or len({case.id for case in cases}) != 4:
        raise ValueError("exactly four unique pinned training inputs required")
    tokenizer = AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"),
                                             local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [tokenizer.apply_chat_template([{"role": "system", "content": case.system},
        {"role": "user", "content": case.prompt}], tokenize=True, add_generation_prompt=True) for case in cases]
    if any(len(prompt) + config["max_new_tokens"] > config["max_model_len"] for prompt in prompts):
        raise ValueError("input exceeds context budget")
    base = (ROOT / manifest["base_cache_files"][0]["source"]).parent
    return config, baseline, base, tokenizer, cases, prompts


def inventory(config, baseline):
    sources = [Path(__file__), ROOT / "configs/rollout_vllm_smoke.json", ROOT / "configs/rollout_cpu_smoke.json",
        ROOT / "configs/cloud_replay_smoke.json",
        ROOT / "scripts/replay_cloud_smoke.py", ROOT / "scripts/verify_cloud_bundle.py",
        ROOT / "scripts/freeze_sft_baseline.py", *(ROOT / "src/agentic_post_training" / name for name in (
            "rollouts/vllm_records.py", "rollouts/records.py", "algorithms/losses.py",
            "rewards/contract.py", "rewards/audit.py", "evaluation/tool_calls.py",
            "environments/calculator.py", "data/preferences.py"))]
    return {"config": config, "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "adapter_sha256": file_hash(baseline / "checkpoint/adapter/adapter_model.safetensors"),
        "data_manifest_sha256": file_hash(ROOT / "data/preferences/dpo_v2/manifest.json"),
        "training_updates": 0, "development_inputs_sampled": False,
        "probability_semantics": "raw_logprobs; behavior equals policy only at T=1 without filters or penalties"}


def check_gpu(config):
    import torch
    actual = {name: version(name) for name in config["versions"]}
    if actual != config["versions"] or not torch.cuda.is_available():
        raise ValueError(f"expected fixed CUDA dependencies, got {actual}")
    return torch.cuda.get_device_name(0)


def check_prior_replay():
    path = ROOT / "artifacts/evaluations/cloud-replay-smoke/replay.json"
    gate = read(path)
    if gate["contract"] != read(ROOT / "configs/cloud_replay_smoke.json"):
        raise ValueError("prior replay contract differs")
    if not gate["single_sequence_probability_check_passed"] or not gate["probabilities_replayed"]:
        raise ValueError("prior single-sequence replay must pass first")
    for name, digest in gate["input_sha256"].items():
        source = ROOT / "artifacts/trajectories/cloud-model-smoke" / f"{name}.json"
        if file_hash(source) != digest:
            raise ValueError("prior smoke artifact changed")
        comparison = compare_logps(read(source)["raw_policy_token_log_probs"],
            gate["replayed_policy_token_log_probs"], gate["contract"]["absolute_logp_tolerance"])
        if comparison != gate["comparisons"][name] or not comparison["passed"]:
            raise ValueError("prior replay result cannot be reproduced from saved scores")
    return path


def effective_config(engine):
    c = engine.llm_engine.get_vllm_config()
    return {"dtype": str(c.model_config.dtype), "logprobs_mode": c.model_config.logprobs_mode,
        "max_model_len": c.model_config.max_model_len, "enforce_eager": c.model_config.enforce_eager,
        "enable_prefix_caching": c.cache_config.enable_prefix_caching,
        "enable_chunked_prefill": c.scheduler_config.chunked_prefill_enabled,
        "max_num_seqs": c.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": c.scheduler_config.max_num_batched_tokens,
        "max_lora_rank": c.lora_config.max_lora_rank}


def generate(output):
    config, baseline, base, tokenizer, cases, prompts = context()
    device = check_gpu(config)
    prior = check_prior_replay()
    output.mkdir(parents=True, exist_ok=False)
    inv = inventory(config, baseline)
    inv.update(device=device, prerequisite_replay_sha256=file_hash(prior))
    write(output / "run_manifest.json", inv)
    shutil.copyfile(prior, output / "prerequisite_replay.json")
    for relative in inv["source_sha256"]:
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch.distributed as dist
    engine = LLM(model=str(base), tokenizer=str(baseline / "checkpoint/tokenizer"),
        dtype="half", trust_remote_code=False, enable_lora=True, max_lora_rank=8, max_loras=1,
        max_model_len=config["max_model_len"], max_num_seqs=config["max_num_seqs"],
        max_num_batched_tokens=config["max_num_batched_tokens"],
        gpu_memory_utilization=config["gpu_memory_utilization"], enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, seed=config["base_seed"],
        generation_config="vllm", logprobs_mode="raw_logprobs")
    checks = {"effective_engine": effective_config(engine), "device": device}
    if checks["effective_engine"]["logprobs_mode"] != "raw_logprobs":
        raise ValueError("engine must return raw policy logprobs")
    shared = dict(n=1, temperature=1, top_p=1, top_k=-1, max_tokens=64,
        min_tokens=0, repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
        logprobs=1, stop_token_ids=[tokenizer.eos_token_id], ignore_eos=False, detokenize=False)
    checks["sampling_parameters"] = shared
    plan = [(g, i, config["base_seed"] + g * config["seed_group_stride"] + i)
            for g in range(4) for i in range(4)]
    params = [SamplingParams(seed=seed, **shared) for g, i, seed in plan]
    request = LoRARequest("sft-v2-phase3", 1, str(baseline / "checkpoint/adapter"))
    try:
        started = time.perf_counter()
        outputs = engine.generate([{"prompt_token_ids": prompts[g]} for g, i, seed in plan],
                                  params, lora_request=request, use_tqdm=False)
        checks["generation_call_seconds"] = time.perf_counter() - started
        if len(outputs) != len(plan):
            raise ValueError("missing sampling outputs")
        records = []
        with (output / "rollouts.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for result, (g, i, seed) in zip(outputs, plan):
                sample = sample_from_output(result, prompts[g], cases[g], i, seed, tokenizer, config)
                reward = score_reward(sample.text, cases[g], finish_reason=sample.finish_reason)
                row = {"sample": asdict(sample), "case": asdict(cases[g]), "score": reward["score"],
                       "semantic_pass": reward["semantic_pass"]}
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"{cases[g].id} sample {i}: {sample.finish_reason}; reward={reward['reward']}", flush=True)
        # One extra call probes same-engine reproducibility; not a seventeenth training sample.
        probe_output = engine.generate([{"prompt_token_ids": prompts[0]}], params[0],
                                      lora_request=request, use_tqdm=False)[0]
        probe = sample_from_output(probe_output, prompts[0], cases[0], 0, plan[0][2], tokenizer, config)
        first = RolloutSample(**records[0]["sample"])
        checks["reproduction_probe"] = asdict(probe)
        checks["probe_tokens_equal"] = probe.response_token_ids == first.response_token_ids
        checks["probe_logp_comparison"] = (compare_logps(first.policy_token_log_probs,
            probe.policy_token_log_probs, config["absolute_logp_tolerance"])
            if len(first.response_token_ids) == len(probe.response_token_ids)
            else {"passed": False, "reason": "different_length"})
    finally:
        # Explicitly terminate v1 workers before a separate HF scorer process starts.
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
    checks["engine_shutdown_called"] = True
    write(output / "generation_checks.json", checks)
    print(json.dumps({"generation_saved": 16, "extra_reproduction_calls": 1,
        "probe_tokens_equal": checks["probe_tokens_equal"], "output": str(output)}), flush=True)


def load_and_validate(output):
    config, baseline, base, tokenizer, cases, prompts = context()
    inv = read(output / "run_manifest.json")
    expected = inventory(config, baseline)
    if any(inv[key] != value for key, value in expected.items()):
        raise ValueError("run model/data/config/source identity differs")
    for relative, digest in inv["source_sha256"].items():
        if file_hash(output / "source" / relative) != digest:
            raise ValueError("archived source differs")
    if file_hash(output / "prerequisite_replay.json") != inv["prerequisite_replay_sha256"]:
        raise ValueError("prerequisite snapshot changed")
    rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(rows) != 16:
        raise ValueError("expected all 16 samples, including failures/duplicates")
    groups = []
    for g, case in enumerate(cases):
        group = []
        for i, row in enumerate(rows[g * 4:(g + 1) * 4]):
            sample = RolloutSample(**row["sample"])
            if (row["case"] != asdict(case) or sample.group_id != case.id or sample.case_id != case.id
                or sample.sample_index != i or sample.seed != 42 + g * 1000 + i
                or sample.prompt_token_ids != prompts[g]
                or len(sample.response_token_ids) > 64
                or (sample.finish_reason == "length" and len(sample.response_token_ids) != 64)
                or sample.policy_token_log_probs != sample.behavior_token_log_probs
                or sample.eos_token_id != tokenizer.eos_token_id
                or sample.text != tokenizer.decode(sample.response_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                or sample.raw_text != tokenizer.decode(sample.response_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)):
                raise ValueError("rollout identity, decoding, sampling or length mismatch")
            group.append(sample)
        groups.append(group)
    packed = pack_rollout_groups(groups, tokenizer.pad_token_id)
    rewards = audit_records(rows)
    return config, baseline, base, tokenizer, rows, groups, packed, rewards


def replay(output):
    if any((output / name).exists() for name in ("replay_checks.json", "response_batch.pt", "reward_audit.json", "summary.json")):
        raise FileExistsError("replay artifacts already exist; refusing to overwrite")
    config, baseline, base, tokenizer, rows, groups, packed, rewards = load_and_validate(output)
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    device = check_gpu(config)
    if device != read(output / "run_manifest.json")["device"]:
        raise ValueError("generation and scoring GPU model differ")
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float16, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(baseline / "checkpoint/adapter"),
        is_trainable=False, autocast_adapter_dtype=False).to("cuda").eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("scoring model must be frozen")
    before = {name: tensor._version for name, tensor in model.named_parameters()}
    records = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for group in groups:
            for sample in group:
                ids = torch.tensor([sample.prompt_token_ids + sample.response_token_ids], device="cuda")
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                logps = response_logps(logits, ids, len(sample.prompt_token_ids)).cpu().tolist()
                records.append({"group_id": sample.group_id, "sample_index": sample.sample_index,
                    "response_token_ids": sample.response_token_ids, "replayed_policy_log_probs": logps,
                    "comparison": compare_logps(sample.policy_token_log_probs, logps,
                                                config["absolute_logp_tolerance"])})
    torch.cuda.synchronize()
    checks = {"records": records, "forward_and_extraction_seconds": time.perf_counter() - started,
        "parameter_versions_unchanged": before == {name: p._version for name, p in model.named_parameters()},
        "frozen_no_grad": all(not p.requires_grad and p.grad is None for p in model.parameters()),
        "attention": "eager", "dtype": "float16", "log_softmax_dtype": "float32", "use_cache": False}
    write(output / "replay_checks.json", checks)
    torch.save(packed, output / "response_batch.pt")
    write(output / "reward_audit.json", rewards)
    summary = summarize(output, config, rows, groups, packed, rewards)
    write(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["rollout_interface_checks_passed"]:
        raise SystemExit(2)


def summarize(output, config, rows, groups, packed, rewards):
    generation = read(output / "generation_checks.json")
    replayed = read(output / "replay_checks.json")
    if len(replayed["records"]) != 16:
        raise ValueError("incomplete replay records")
    for sample, record in zip([s for group in groups for s in group], replayed["records"]):
        if (record["group_id"] != sample.group_id or record["sample_index"] != sample.sample_index
            or record["response_token_ids"] != sample.response_token_ids
            or record["comparison"] != compare_logps(sample.policy_token_log_probs,
                record["replayed_policy_log_probs"], config["absolute_logp_tolerance"])):
            raise ValueError("saved replay mapping or comparison differs")
    probe = RolloutSample(**generation["reproduction_probe"])
    first = groups[0][0]
    if (probe.prompt_token_ids != first.prompt_token_ids or probe.group_id != first.group_id
        or probe.case_id != first.case_id or probe.sample_index != 0 or probe.seed != first.seed):
        raise ValueError("probe identity differs")
    probe_equal = probe.response_token_ids == first.response_token_ids
    probe_comparison = (compare_logps(first.policy_token_log_probs, probe.policy_token_log_probs,
                                     config["absolute_logp_tolerance"])
                        if len(first.response_token_ids) == len(probe.response_token_ids)
                        else {"passed": False, "reason": "different_length"})
    if generation["probe_tokens_equal"] != probe_equal or generation["probe_logp_comparison"] != probe_comparison:
        raise ValueError("saved reproduction result differs")
    effective = generation["effective_engine"]
    if (effective["dtype"] != "torch.float16" or effective["logprobs_mode"] != "raw_logprobs"
        or not effective["enforce_eager"] or effective["enable_prefix_caching"]
        or effective["max_model_len"] != 512 or effective["max_num_seqs"] != 4
        or effective["max_num_batched_tokens"] != 512 or effective["max_lora_rank"] != 8
        or not effective["enable_chunked_prefill"]):
        raise ValueError("unsupported effective engine configuration")
    expected_sampling = dict(n=1, temperature=1, top_p=1, top_k=-1, max_tokens=64,
        min_tokens=0, repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
        logprobs=1, stop_token_ids=[first.eos_token_id], ignore_eos=False, detokenize=False)
    if generation["sampling_parameters"] != expected_sampling:
        raise ValueError("saved sampling distribution differs")
    numeric = all(row["comparison"]["passed"] for row in replayed["records"])
    passed = (numeric and probe_equal and probe_comparison["passed"] and generation["engine_shutdown_called"]
              and replayed["frozen_no_grad"] and replayed["parameter_versions_unchanged"])
    return {"samples": 16, "extra_reproduction_calls": 1, "groups": [
        {"group_id": group[0].group_id, "unique_texts": len({s.text for s in group}),
         "unique_token_sequences": len({tuple(s.response_token_ids) for s in group}),
         "lengths": [len(s.response_token_ids) for s in group],
         "finish_reasons": dict(Counter(s.finish_reason for s in group)),
         "strict_pass": sum(row["score"]["end_to_end_pass"] for row in rows[g*4:(g+1)*4]),
         "semantic_pass": sum(row["semantic_pass"] for row in rows[g*4:(g+1)*4]),
         "reward_group": rewards["groups"][g]} for g, group in enumerate(groups)],
        "strict_pass": sum(row["score"]["end_to_end_pass"] for row in rows),
        "semantic_pass": sum(row["semantic_pass"] for row in rows),
        "invalid_json": sum(not row["score"]["json_valid"] for row in rows),
        "eligible_samples": rewards["eligible_samples"], "reward_sum": rewards["reward_sum"],
        "eligible_groups": rewards["eligible_groups"], "zero_variance_groups": rewards["zero_variance_groups"],
        "zero_variance_rate": rewards["zero_variance_rate"],
        "has_reward_learning_signal": rewards["has_reward_learning_signal"],
        "max_replay_abs_error": max(row["comparison"]["max_abs_error"] for row in replayed["records"]),
        "replay_failed_samples": sum(not row["comparison"]["passed"] for row in replayed["records"]),
        "absolute_logp_tolerance": config["absolute_logp_tolerance"],
        "response_batch_shape": list(packed["response_mask"].shape),
        "response_tokens": int(packed["response_mask"].sum()),
        "padding_positions": packed["response_mask"].numel() - int(packed["response_mask"].sum()),
        "eos_tokens_retained": sum(s.finish_reason == "eos" for group in groups for s in group),
        "effective_engine": effective, "first_sample_reproduced": probe_equal and probe_comparison["passed"],
        "rollout_interface_checks_passed": passed, "training_updates": 0, "training_admitted": False,
        "generation_call_seconds": generation["generation_call_seconds"],
        "replay_seconds": replayed["forward_and_extraction_seconds"],
        "files_sha256": {name: file_hash(output / name) for name in ("run_manifest.json", "rollouts.jsonl",
            "generation_checks.json", "replay_checks.json", "reward_audit.json", "response_batch.pt", "prerequisite_replay.json")}}


def audit(output):
    import torch
    config, baseline, base, tokenizer, rows, groups, packed, rewards = load_and_validate(output)
    saved = torch.load(output / "response_batch.pt", map_location="cpu", weights_only=True)
    if set(saved) != set(packed) or any(not torch.equal(saved[key], value) for key, value in packed.items()):
        raise ValueError("saved response tensors or length masks differ")
    if read(output / "reward_audit.json") != rewards:
        raise ValueError("saved rewards or advantages differ")
    summary = summarize(output, config, rows, groups, packed, rewards)
    if summary != read(output / "summary.json"):
        raise ValueError("saved summary differs")
    print(json.dumps({"artifact_audit_passed": True, "samples_reaudited": 16,
        "model_loaded": False, "policy_probabilities_recomputed_during_audit": False,
        "rollout_interface_checks_passed": summary["rollout_interface_checks_passed"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("run", "generate", "replay", "audit"), default="run")
    parser.add_argument("--output", type=Path,
        default=ROOT / "artifacts/trajectories/rollout-vllm-phase4-20261020-01")
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", VLLM_NO_USAGE_STATS="1")
    if args.stage == "run":
        if args.output.exists():
            raise FileExistsError(args.output)
        for stage in ("generate", "replay", "audit"):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                "--stage", stage, "--output", str(args.output.resolve())], cwd=ROOT)
            if result.returncode:
                raise SystemExit(result.returncode)
    elif args.stage == "generate":
        generate(args.output)
    elif args.stage == "replay":
        replay(args.output)
    else:
        audit(args.output)


if __name__ == "__main__":
    main()
