"""Teacher-force saved native IDs; no sampling and no policy update."""

import argparse
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def compare_logps(saved, replayed, tolerance):
    if not saved or len(saved) != len(replayed):
        raise ValueError("non-empty aligned probabilities required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("finite positive tolerance required")
    if any(not math.isfinite(x) or x > 1e-6 for x in [*saved, *replayed]):
        raise ValueError("invalid log probability")
    errors = [abs(a - b) for a, b in zip(saved, replayed)]
    return {"max_abs_error": max(errors), "mean_abs_error": sum(errors) / len(errors),
            "per_token_abs_error": errors, "saved_sequence_logp": sum(saved),
            "replayed_sequence_logp": sum(replayed),
            "sequence_logp_abs_error": abs(sum(saved) - sum(replayed)),
            "passed": max(errors) <= tolerance}


def response_logps(logits, ids, prompt_length):
    from agentic_post_training.algorithms.losses import causal_token_log_probs
    if ids.ndim != 2 or ids.shape[0] != 1 or not 1 <= prompt_length < ids.shape[1]:
        raise ValueError("single complete sequence and non-empty response required")
    # Logit t predicts token t+1; the first response is predicted at prompt_length-1.
    return causal_token_log_probs(logits.float(), ids)[0, prompt_length - 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "artifacts/trajectories/cloud-model-smoke")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluations/cloud-replay-smoke/replay.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from verify_cloud_bundle import main as verify_bundle, file_hash
    from cloud_model_smoke import validate_response
    verify_bundle()
    config_path = ROOT / "configs/cloud_replay_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["dtype"] != "float16" or config["log_softmax_dtype"] != "float32":
        raise ValueError("unsupported scoring precision")
    inputs = {name: args.input_dir / f"{name}.json" for name in ("hf", "vllm")}
    if any(file_hash(path) != config["input_sha256"][name] for name, path in inputs.items()):
        raise ValueError("inputs differ from pinned smoke records")
    rows = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in inputs.items()}
    hf, vllm = rows["hf"], rows["vllm"]
    for key in ("versions", "device", "dtype", "case", "prompt_token_ids", "response_token_ids",
                "text", "finish_reason", "source_sha256", "baseline_manifest_sha256", "adapter_sha256"):
        if hf[key] != vllm[key]:
            raise ValueError(f"backend identity/sequence differs: {key}")
    if hf["backend"] != "hf" or vllm["backend"] != "vllm" or hf["dtype"] != config["dtype"]:
        raise ValueError("backend or dtype mismatch")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from agentic_post_training.evaluation.tool_calls import EvalCase, score_output
    versions = {name: version(name) for name in hf["versions"]}
    if versions != hf["versions"] or not torch.cuda.is_available():
        raise ValueError("same CUDA dependencies required")
    if torch.cuda.get_device_name(0) != hf["device"]:
        raise ValueError("this calibration is pinned to the smoke GPU model")
    baseline = ROOT / "artifacts/baselines/sft-v2-phase3"
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    adapter = baseline / "checkpoint/adapter"
    for actual, expected in ((file_hash(baseline / "manifest.json"), hf["baseline_manifest_sha256"]),
        (file_hash(adapter / "adapter_model.safetensors"), hf["adapter_sha256"]),
        (file_hash(ROOT / "scripts/cloud_model_smoke.py"), hf["source_sha256"])):
        if actual != expected:
            raise ValueError("local model or generation source identity differs")
    tokenizer = AutoTokenizer.from_pretrained(str(baseline / "checkpoint/tokenizer"), local_files_only=True,
                                             trust_remote_code=False)
    case = EvalCase(**hf["case"])
    prompt, response = hf["prompt_token_ids"], hf["response_token_ids"]
    encoded = tokenizer.apply_chat_template([{"role": "system", "content": case.system},
        {"role": "user", "content": case.prompt}], tokenize=True, add_generation_prompt=True)
    if encoded != prompt or tokenizer.decode(response, skip_special_tokens=True,
        clean_up_tokenization_spaces=False) != hf["text"]:
        raise ValueError("native token/text alignment differs")
    for row in rows.values():
        validate_response(prompt, response, row["raw_policy_token_log_probs"],
                          tokenizer.eos_token_id, row["finish_reason"])
        if score_output(row["text"], case) != row["score"]:
            raise ValueError("saved evaluation differs")
    base = (ROOT / manifest["base_cache_files"][0]["source"]).parent
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float16, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
        autocast_adapter_dtype=False).to("cuda").eval()
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("replay model must be frozen")
    ids = torch.tensor([prompt + response], dtype=torch.long, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        replayed = response_logps(logits, ids, len(prompt)).cpu().tolist()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    comparisons = {name: compare_logps(row["raw_policy_token_log_probs"], replayed,
        config["absolute_logp_tolerance"]) for name, row in rows.items()}
    passed = all(row["passed"] for row in comparisons.values())
    report = {"contract": config, "input_sha256": config["input_sha256"], "versions": versions,
        "device": torch.cuda.get_device_name(0), "source_sha256": {
            str(path.relative_to(ROOT)): file_hash(path) for path in (
                Path(__file__), config_path, ROOT / "src/agentic_post_training/algorithms/losses.py")},
        "baseline_manifest_sha256": hf["baseline_manifest_sha256"], "adapter_sha256": hf["adapter_sha256"],
        "prompt_token_ids": prompt, "response_token_ids": response,
        "replayed_policy_token_log_probs": replayed, "comparisons": comparisons,
        "unique_sequences_scored": 1, "probabilities_replayed": True,
        "single_sequence_probability_check_passed": passed, "training_admitted": False,
        "sampling_calls": 0, "training_updates": 0, "forward_and_extraction_seconds": elapsed,
        "use_cache": False, "attention": "eager", "log_softmax_dtype": "float32"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"single_sequence_probability_check_passed": passed,
        "absolute_logp_tolerance": config["absolute_logp_tolerance"],
        "comparisons": {name: {key: value for key, value in row.items() if key != "per_token_abs_error"}
                        for name, row in comparisons.items()}, "output": str(args.output),
        "training_admitted": False}, ensure_ascii=False))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
