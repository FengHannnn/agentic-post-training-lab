"""One frozen-SFT GPU prompt per backend; full rollout calibration is later."""

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def validate_response(prompt, response, logps, eos, finish):
    if not response or len(response) != len(logps):
        raise ValueError("response IDs and log probabilities must align")
    if any(not math.isfinite(p) or p > 1e-6 for p in logps):
        raise ValueError("invalid token log probability")
    if eos in response[:-1]:
        raise ValueError("tokens after EOS")
    if finish not in ("eos", "length") or ((response[-1] == eos) != (finish == "eos")):
        raise ValueError("EOS and finish reason mismatch")
    if not prompt or len(response) > 64:
        raise ValueError("invalid prompt or smoke length")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hf", "vllm"), required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    from verify_cloud_bundle import main as verify_bundle, file_hash
    verify_bundle()
    output = ROOT / "artifacts/trajectories/cloud-model-smoke" / f"{args.backend}.json"
    if output.exists():
        raise FileExistsError(output)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    from agentic_post_training.evaluation.tool_calls import EvalCase, score_output
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    expected = {"torch": "2.8.0", "vllm": "0.11.0", "transformers": "4.57.6", "peft": "0.17.1"}
    versions = {name: version(name) for name in expected}
    if any(versions[name].split("+")[0] != value for name, value in expected.items()):
        raise ValueError(f"unexpected dependency versions: {versions}")

    baseline = ROOT / "artifacts/baselines/sft-v2-phase3"
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    base = (ROOT / manifest["base_cache_files"][0]["source"]).parent
    adapter = baseline / "checkpoint/adapter"
    tokenizer_path = baseline / "checkpoint/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = json.loads((ROOT / "configs/rollout_cpu_smoke.json").read_text(encoding="utf-8"))
    # Reuse the first saved training-input target, after the bundle identity check.
    historical = ROOT / "artifacts/trajectories/rollout-smoke-phase4-20261017-01/rollouts.jsonl"
    case = EvalCase(**json.loads(historical.read_text(encoding="utf-8").splitlines()[0])["case"])
    if case.id != config["case_ids"][0]:
        raise ValueError("smoke target differs from pinned training input")
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": case.system}, {"role": "user", "content": case.prompt}
    ], tokenize=True, add_generation_prompt=True)
    if len(prompt) + 64 > 512:
        raise ValueError("prompt exceeds smoke context")
    eos = tokenizer.eos_token_id
    if type(eos) is not int:
        raise ValueError("single tokenizer EOS required")
    if args.backend == "hf":
        from peft import PeftModel
        model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.float16, attn_implementation="eager")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False,
            autocast_adapter_dtype=False).to("cuda").eval()
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("smoke model must be frozen")
        ids = torch.tensor([prompt], device="cuda")
        generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=64,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
            use_cache=True, return_dict_in_generate=True, output_logits=True)
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                generation_config=generation, use_model_defaults=False)
        response = generated.sequences[0, len(prompt):].tolist()
        logps = [torch.log_softmax(logits[0].float(), -1)[token].item()
                 for token, logits in zip(response, generated.logits)]
        finish = "eos" if response and response[-1] == eos else "length"
    else:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        engine = LLM(model=str(base), tokenizer=str(tokenizer_path), trust_remote_code=False,
            dtype="half", enable_lora=True, max_lora_rank=8, max_loras=1,
            max_model_len=512, max_num_seqs=1, max_num_batched_tokens=512,
            gpu_memory_utilization=0.25, enforce_eager=True, seed=42,
            enable_prefix_caching=False, enable_chunked_prefill=False,
            generation_config="vllm", logprobs_mode="raw_logprobs")
        params = SamplingParams(temperature=0, top_p=1, top_k=-1, max_tokens=64,
            repetition_penalty=1, presence_penalty=0, frequency_penalty=0,
            seed=42, logprobs=1, stop_token_ids=[eos], ignore_eos=False,
            detokenize=False)
        requests = engine.generate([{"prompt_token_ids": prompt}], params,
            lora_request=LoRARequest("sft-v2-phase3", 1, str(adapter)), use_tqdm=False)
        if len(requests) != 1 or list(requests[0].prompt_token_ids) != prompt:
            raise ValueError("engine prompt IDs differ")
        completion = requests[0].outputs[0]
        response = list(completion.token_ids)
        if completion.logprobs is None or len(completion.logprobs) != len(response):
            raise ValueError("missing engine probabilities")
        logps = [values[token].logprob for token, values in zip(response, completion.logprobs)]
        finish = "eos" if completion.finish_reason == "stop" and response and response[-1] == eos else completion.finish_reason
    validate_response(prompt, response, logps, eos, finish)
    text = tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    result = {"backend": args.backend, "versions": versions, "device": torch.cuda.get_device_name(0),
        "dtype": "float16", "case": asdict(case), "prompt_token_ids": prompt,
        "response_token_ids": response, "raw_policy_token_log_probs": logps,
        "finish_reason": finish, "text": text, "score": score_output(text, case),
        "sampling": "greedy; this is NOT stochastic rollout or a throughput benchmark",
        "probabilities_replayed": False, "training_admitted": False,
        "source_sha256": file_hash(Path(__file__)),
        "baseline_manifest_sha256": file_hash(baseline / "manifest.json"),
        "adapter_sha256": file_hash(adapter / "adapter_model.safetensors")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"backend": args.backend, "text": text, "finish_reason": finish,
        "response_tokens": len(response), "strict_pass": result["score"]["end_to_end_pass"],
        "output": str(output), "probabilities_replayed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
