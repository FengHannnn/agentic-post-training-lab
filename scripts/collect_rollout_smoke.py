"""Small frozen-SFT stochastic rollout, token replay and model-free artifact audit."""

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import torch

from compare_sft_dpo import file_hash, write_json
from freeze_sft_baseline import verify
from agentic_post_training.data.hf_tokenizer import HFChatTokenizer
from agentic_post_training.data.preferences import load_preference_splits, semantic_pass
from agentic_post_training.evaluation.tool_calls import score_output
from agentic_post_training.rollouts.hf_sampling import sample_response
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.trainers.dpo import parameter_digest

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same_sample(left, right):
    a, b = asdict(left), asdict(right)
    fields = ("policy_token_log_probs", "behavior_token_log_probs")
    return (all(value == b[key] for key, value in a.items() if key not in fields)
        and all(torch.allclose(torch.tensor(a[key]), torch.tensor(b[key]), atol=1e-6, rtol=1e-6) for key in fields))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="4题×4随机回答的CPU rollout验证，不训练")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rollout_cpu_smoke.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only and args.output is None:
        parser.error("audit-only requires output")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    config = read_json(args.config)
    if (config["group_size"] < 2 or len(config["case_ids"])*config["group_size"] > 16
        or len(set(config["case_ids"])) != len(config["case_ids"]) or not config["case_ids"]
        or not 0 < config["max_new_tokens"] <= 64 or config["cpu_threads"] <= 0
        or config["top_k"] != 0 or config["seed_group_stride"] < config["group_size"]
        or config["reproduction_probes"] != 1):
        raise ValueError("configuration exceeds the small rollout scope")
    torch.set_num_threads(config["cpu_threads"])
    baseline = ROOT / config["baseline"]
    verify(baseline)
    manifest = read_json(baseline / "manifest.json")
    folder = ROOT / config["preference_data"]
    data_manifest = read_json(folder / "manifest.json")
    if data_manifest["baseline_manifest_sha256"] != file_hash(baseline / "manifest.json"):
        raise ValueError("data and baseline differ")
    train = load_preference_splits(folder, data_manifest)["train"]
    indexed = {row.id: row for row in train}
    if any(identifier not in indexed for identifier in config["case_ids"]):
        raise ValueError("all sampling inputs must come from the pinned training split")
    cases = [indexed[identifier].target() for identifier in config["case_ids"]]
    tokenizer = HFChatTokenizer(AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"),
        local_files_only=True, trust_remote_code=False))
    sources = [Path(__file__), ROOT / "src/agentic_post_training/rollouts/records.py",
        ROOT / "src/agentic_post_training/rollouts/hf_sampling.py",
        ROOT / "src/agentic_post_training/algorithms/losses.py", ROOT / "src/agentic_post_training/data/hf_tokenizer.py",
        ROOT / "src/agentic_post_training/evaluation/tool_calls.py"]
    inventory = {"config": config, "config_sha256": file_hash(args.config),
        "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "adapter_sha256": file_hash(baseline / "checkpoint/adapter/adapter_model.safetensors"),
        "data_manifest_sha256": file_hash(folder / "manifest.json"),
        "model": manifest["model"], "resolved_revision": manifest["resolved_revision"],
        "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in sources},
        "versions": {name: version(name) for name in ("torch", "transformers", "peft")},
        "runtime": {"platform": platform.platform(), "python": platform.python_version(), "device": "cpu", "dtype": "float32", "attention": "eager"},
        "sampling": {"do_sample": True, "top_k": 0, "temperature": config["temperature"], "top_p": config["top_p"],
            "repetition_penalty": 1.0, "use_model_defaults": False, "num_return_sequences_per_call": 1},
        "training_updates": 0, "development_inputs_sampled": False,
        "probability_semantics": {"policy": "log_softmax of unprocessed logits at emitted IDs",
            "behavior": "log_softmax of processed generation scores at emitted IDs"}}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = args.output.resolve() if args.output else ROOT / "artifacts/trajectories" / f"rollout-smoke-{stamp}-{uuid4().hex[:8]}"
    if args.audit_only:
        if read_json(output / "run_manifest.json") != inventory:
            raise ValueError("rollout inventory differs")
        records = [json.loads(line) for line in (output / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
        checks = read_json(output / "run_checks.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "run_manifest.json", inventory)
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(manifest["model"], revision=manifest["resolved_revision"],
            cache_dir=str(ROOT / ".cache/huggingface"), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="eager").cpu()
        model = PeftModel.from_pretrained(base, str(baseline / "checkpoint/adapter"), is_trainable=False)
        model.requires_grad_(False)
        model.eval()
        initial = parameter_digest(model)
        records = []
        print(json.dumps({"output": str(output), "samples": len(cases)*config["group_size"]}, ensure_ascii=False), flush=True)
        with (output / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for group_index, case in enumerate(cases):
                for sample_index in range(config["group_size"]):
                    seed = config["base_seed"] + group_index*config["seed_group_stride"] + sample_index
                    started = time.perf_counter()
                    sample, error = sample_response(model, tokenizer, case, group_id=case.id, sample_index=sample_index,
                        seed=seed, max_new_tokens=config["max_new_tokens"], temperature=config["temperature"], top_p=config["top_p"])
                    seconds = time.perf_counter()-started
                    score = score_output(sample.text, case)
                    row = {"sample": asdict(sample), "case": asdict(case), "score": score,
                        "semantic_pass": semantic_pass(score, case.expected_type),
                        "policy_replay_max_abs_error": error, "sample_and_replay_seconds": seconds}
                    records.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                    stream.flush()
                    print(f"{case.id} sample {sample_index+1}: {score['failure'] or 'pass'}; {sample.finish_reason}", flush=True)
        first = RolloutSample(**records[0]["sample"])
        probe, error = sample_response(model, tokenizer, cases[0], group_id=first.group_id, sample_index=0,
            seed=first.seed, max_new_tokens=config["max_new_tokens"], temperature=config["temperature"], top_p=config["top_p"])
        checks = {"initial_parameter_sha256": initial, "final_parameter_sha256": parameter_digest(model),
            "policy_frozen_no_grad": all(not value.requires_grad and value.grad is None for value in model.parameters()),
            "reproduction_probe": asdict(probe), "probe_replay_max_abs_error": error,
            "first_sample_reproduced": same_sample(first, probe)}
        write_json(output / "run_checks.json", checks)
    expected_count = len(cases)*config["group_size"]
    if len(records) != expected_count:
        raise ValueError("rollout collection is incomplete")
    groups, observations = [], []
    for group_index, case in enumerate(cases):
        group, selected = [], records[group_index*config["group_size"]:(group_index+1)*config["group_size"]]
        prompt_ids = tokenizer.apply_chat_template([{"role": "system", "content": case.system},
            {"role": "user", "content": case.prompt}], tokenize=True, add_generation_prompt=True)
        for sample_index, row in enumerate(selected):
            sample = RolloutSample(**row["sample"])
            score = score_output(sample.text, case)
            if (sample.case_id != case.id or sample.group_id != case.id or sample.sample_index != sample_index
                or sample.seed != config["base_seed"]+group_index*config["seed_group_stride"]+sample_index
                or sample.prompt_token_ids != prompt_ids or row["case"] != asdict(case) or row["score"] != score
                or row["semantic_pass"] != semantic_pass(score, case.expected_type)
                or sample.raw_text != tokenizer.decode(sample.response_token_ids)
                or sample.text != tokenizer.raw.decode(sample.response_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                or len(sample.response_token_ids) > config["max_new_tokens"]
                or (sample.finish_reason == "length" and len(sample.response_token_ids) != config["max_new_tokens"])):
                raise ValueError("saved rollout, token decoding or score differs")
            if not math.isfinite(row["policy_replay_max_abs_error"]) or row["policy_replay_max_abs_error"] > 1e-4*(1+max(abs(value) for value in sample.policy_token_log_probs)):
                raise ValueError("invalid replay check")
            if config["temperature"] == 1 and config["top_p"] == 1 and not torch.allclose(
                torch.tensor(sample.policy_token_log_probs), torch.tensor(sample.behavior_token_log_probs), atol=1e-6, rtol=1e-6):
                raise ValueError("unmodified sampling distribution differs from policy")
            group.append(sample)
        groups.append(group)
        observations.append({"case_id": case.id, "unique_texts": len({sample.text for sample in group}),
            "unique_token_sequences": len({tuple(sample.response_token_ids) for sample in group}),
            "strict_pass": sum(row["score"]["end_to_end_pass"] for row in selected),
            "semantic_pass": sum(row["semantic_pass"] for row in selected),
            "strict_outcomes_constant": len({row["score"]["end_to_end_pass"] for row in selected}) == 1,
            "lengths": [len(sample.response_token_ids) for sample in group],
            "finish_reasons": dict(Counter(sample.finish_reason for sample in group))})
    probe = RolloutSample(**checks["reproduction_probe"])
    if (checks["initial_parameter_sha256"] != checks["final_parameter_sha256"] or not checks["policy_frozen_no_grad"]
        or not checks["first_sample_reproduced"] or not same_sample(groups[0][0], probe)):
        raise ValueError("frozen policy or reproduction check failed")
    packed = pack_rollout_groups(groups, tokenizer.pad_token_id)
    if args.audit_only:
        saved = torch.load(output / "response_batch.pt", map_location="cpu", weights_only=True)
        if set(saved) != set(packed) or any(not torch.equal(saved[key], value) for key, value in packed.items()):
            raise ValueError("saved masks or packed scores differ")
    else:
        torch.save(packed, output / "response_batch.pt")
    summary = {**inventory, "samples": expected_count, "groups": observations,
        "constant_strict_outcome_groups": sum(row["strict_outcomes_constant"] for row in observations),
        "strict_pass": sum(row["score"]["end_to_end_pass"] for row in records),
        "semantic_pass": sum(row["semantic_pass"] for row in records),
        "max_policy_replay_abs_error": max(row["policy_replay_max_abs_error"] for row in records),
        "response_batch_shape": list(packed["response_mask"].shape),
        "response_tokens": int(packed["response_mask"].sum().item()),
        "eos_tokens_retained": sum(sample.finish_reason == "eos" for group in groups for sample in group),
        "sample_and_replay_seconds": sum(row["sample_and_replay_seconds"] for row in records),
        "policy_unchanged": True, "first_sample_reproduced": True,
        "files_sha256": {name: file_hash(output / name) for name in ("rollouts.jsonl", "run_checks.json", "response_batch.pt")},
        "engine": "transformers_cpu", "vllm_used": False, "grpo_training_implemented": False,
        "limitations": ["4个训练输入的接口验证，不是新性能评测", "3条算术共享数字family，4次采样可能重复",
            "评分沿用既有严格/语义判据，尚未冻结GRPO奖励", "只复现首条采样，不保证跨硬件或跨引擎逐token相同",
            "记录的时间包含teacher-forced重算，不是纯解码吞吐"]}
    if args.audit_only:
        if summary != read_json(output / "summary.json"):
            raise ValueError("saved summary differs")
        write_json(output / "verification.json", {"artifact_audit_passed": True, "samples_reaudited": expected_count,
            "token_ids_decoding_scores_and_masks_recomputed": True, "model_weights_loaded": False,
            "policy_probabilities_recomputed_during_audit": False})
    else:
        write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "strict_pass": summary["strict_pass"], "samples": expected_count,
        "max_replay_error": summary["max_policy_replay_abs_error"], "constant_groups": summary["constant_strict_outcome_groups"],
        "training_updates": 0, "audit_only": args.audit_only}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
