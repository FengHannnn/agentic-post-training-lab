"""One bounded cloud run: rollout, warm benchmark, FP32 scoring, CPU audit, ZIP."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from collect_vllm_rollout import (ROOT, context, read, write, check_gpu, effective_config, inventory)
from verify_cloud_bundle import file_hash
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.closeout import select_cases, sample_seed, validate_records, benchmark_summary, budget_limits
from agentic_post_training.rollouts.records import RolloutSample
from agentic_post_training.rollouts.trainer_scores import score_response, no_update_metrics, pack_trainer_scores
from agentic_post_training.rollouts.vllm_records import sample_from_output
from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss


def setup():
    config, baseline, base, tokenizer, _, _ = context()
    closeout = read(ROOT / "configs/phase4_closeout.json")
    if (closeout["contract_version"] != "phase4-closeout-48x4-v1" or closeout["group_size"] != 4
        or closeout["benchmark_samples"] != 16 or closeout["generation_requests"] != 224
        or closeout["max_wall_seconds"] != 1800 or closeout["max_run_budget_yuan"] != 10
        or closeout["old_new_logp_tolerance"] != 1e-5 or closeout["training_admitted"]):
        raise ValueError("fixed closeout contract differs")
    prior = ROOT / "artifacts/evaluations/trainer-probability-contract-v1/summary.json"
    if file_hash(prior) != closeout["prior_trainer_summary_sha256"]:
        raise ValueError("prior real trainer check required")
    folder = ROOT / "data/preferences/dpo_v2"
    cases = select_cases(load_preference_splits(folder, read(folder/"manifest.json"))["train"], closeout["case_ids"])
    prompts = [tokenizer.apply_chat_template([{"role": "system", "content": c.system},
        {"role": "user", "content": c.prompt}], tokenize=True, add_generation_prompt=True) for c in cases]
    if any(len(p)+64 > 512 for p in prompts):
        raise ValueError("context budget exceeded")
    return config, closeout, baseline, base, tokenizer, cases, prompts


def identity(config, closeout, baseline):
    inv = inventory(config, baseline)
    inv["closeout_config"] = closeout
    paths = [Path(__file__), ROOT/"configs/phase4_closeout.json", ROOT/"configs/rollout_correction_contract.json",
        ROOT/"configs/trainer_probability_contract.json", ROOT/"src/agentic_post_training/rollouts/closeout.py",
        ROOT/"src/agentic_post_training/rollouts/trainer_scores.py",
        ROOT/"src/agentic_post_training/algorithms/rollout_correction.py",
        ROOT/"src/agentic_post_training/trainers/dpo.py"]
    inv["source_sha256"] = {k.replace('\\','/'): v for k,v in inv["source_sha256"].items()}
    inv["source_sha256"].update({str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in paths})
    return inv


def load_run(output):
    config, closeout, baseline, base, tokenizer, cases, prompts = setup()
    inv = read(output/"run_manifest.json")
    expected = identity(config, closeout, baseline)
    if any(inv[k] != v for k,v in expected.items()):
        raise ValueError("closeout data/model/source identity changed")
    for rel, digest in inv["source_sha256"].items():
        if file_hash(output/"source"/rel) != digest:
            raise ValueError("source snapshot changed")
    rows = [json.loads(x) for x in (output/"rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    groups, packed, rewards = validate_records(rows, cases, prompts, tokenizer, config)
    return config, closeout, baseline, base, tokenizer, rows, groups, packed, rewards


def generate(output):
    config, closeout, baseline, base, tokenizer, cases, prompts = setup()
    device = check_gpu(config)
    output.mkdir(parents=True, exist_ok=False)
    inv = identity(config, closeout, baseline); inv["device"] = device
    write(output/"run_manifest.json", inv)
    for rel in inv["source_sha256"]:
        dest = output/"source"/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch
    import torch.distributed as dist
    started = time.perf_counter()
    engine = LLM(model=str(base), tokenizer=str(baseline/"checkpoint/tokenizer"), dtype="half",
        trust_remote_code=False, enable_lora=True, max_lora_rank=8, max_loras=1,
        max_model_len=512, max_num_seqs=4, max_num_batched_tokens=512,
        gpu_memory_utilization=.25, enforce_eager=True, enable_prefix_caching=False,
        enable_chunked_prefill=True, seed=42, generation_config="vllm", logprobs_mode="raw_logprobs")
    load_seconds = time.perf_counter()-started
    effective = effective_config(engine)
    expected_effective = {"dtype":"torch.float16", "logprobs_mode":"raw_logprobs", "max_model_len":512,
        "enforce_eager":True, "enable_prefix_caching":False, "enable_chunked_prefill":True,
        "max_num_seqs":4, "max_num_batched_tokens":512, "max_lora_rank":8}
    if effective != expected_effective:
        raise ValueError("effective engine settings differ")
    shared = dict(n=1, temperature=1, top_p=1, top_k=-1, max_tokens=64, min_tokens=0,
        repetition_penalty=1, presence_penalty=0, frequency_penalty=0, logprobs=1,
        stop_token_ids=[tokenizer.eos_token_id], ignore_eos=False, detokenize=False)
    plan = [(g,i,sample_seed(g,i)) for g in range(48) for i in range(4)]
    params = [SamplingParams(seed=seed, **shared) for g,i,seed in plan]
    inputs = [{"prompt_token_ids":prompts[g]} for g,i,seed in plan]
    request = LoRARequest("sft-v2-phase3",1,str(baseline/"checkpoint/adapter"))
    def adapt(results, entries):
        if len(results) != len(entries): raise ValueError("missing generation output")
        return [sample_from_output(r,prompts[g],cases[g],i,seed,tokenizer,config)
                for r,(g,i,seed) in zip(results,entries)]
    try:
        torch.cuda.synchronize(); started = time.perf_counter()
        samples = adapt(engine.generate(inputs, params, lora_request=request, use_tqdm=False),plan)
        torch.cuda.synchronize(); generation_seconds=time.perf_counter()-started
        with (output/"rollouts.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
            for sample,(g,i,seed) in zip(samples,plan):
                reward=score_reward(sample.text,cases[g],finish_reason=sample.finish_reason)
                stream.write(json.dumps({"sample":asdict(sample),"case":asdict(cases[g]),
                    "score":reward["score"],"semantic_pass":reward["semantic_pass"]},ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()
        # Both timed modes are warm after the fixed 192-sample run. No retry.
        torch.cuda.synchronize(); started=time.perf_counter()
        batched=adapt(engine.generate(inputs[:16],params[:16],lora_request=request,use_tqdm=False),plan[:16])
        torch.cuda.synchronize(); batch_seconds=time.perf_counter()-started
        serial=[]; started=time.perf_counter()
        for inp,param,entry in zip(inputs[:16],params[:16],plan[:16]):
            serial.extend(adapt(engine.generate([inp],param,lora_request=request,use_tqdm=False),[entry]))
        torch.cuda.synchronize(); serial_seconds=time.perf_counter()-started
        checks={"effective_engine":effective,"sampling_parameters":shared,"engine_load_seconds":load_seconds,
            "generation_seconds":generation_seconds,"generation_requests":224,
            "generation_response_tokens_per_second":sum(len(s.response_token_ids) for s in samples)/generation_seconds,
            "benchmark_batch_samples":[asdict(s) for s in batched],"benchmark_serial_samples":[asdict(s) for s in serial],
            "benchmark":benchmark_summary(batched,serial,batch_seconds,serial_seconds)}
    finally:
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized(): dist.destroy_process_group()
    checks["engine_shutdown_called"]=True
    write(output/"generation_checks.json",checks)
    print(json.dumps({"stage":"generate","samples":192,"benchmark":checks["benchmark"]}),flush=True)


def score(output):
    if any((output/name).exists() for name in ("trainer_scores.jsonl","trainer_checks.json","reward_audit.json","trainer_score_batch.pt")):
        raise FileExistsError("scoring artifacts exist; refusing to overwrite")
    config, closeout, baseline, base, tokenizer, rows, groups, packed, rewards = load_run(output)
    device=check_gpu(config)
    if device != read(output/"run_manifest.json")["device"]: raise ValueError("GPU model changed")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from agentic_post_training.trainers.dpo import parameter_digest
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation="eager")
    model=PeftModel.from_pretrained(model,str(baseline/"checkpoint/adapter"),is_trainable=True,
        autocast_adapter_dtype=False).to("cuda").eval()
    contract=read(ROOT/"configs/trainer_probability_contract.json")
    initial=parameter_digest(model)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (initial!=contract["frozen_fp32_parameter_sha256"]
        or sum(p.numel() for n,p in trainable)!=230400 or any("lora_" not in n for n,p in trainable)
        or any(p.dtype!=torch.float32 for p in model.parameters())
        or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules())):
        raise ValueError("trainer parameter contract differs")
    rw={(r["group_id"],r["sample_index"]):r for r in rewards["records"]}
    records=[]; torch.cuda.synchronize(); started=time.perf_counter()
    with (output/"trainer_scores.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
        for group in groups:
            for s in group:
                old=score_response(model,s.prompt_token_ids,s.response_token_ids,grad_enabled=False)
                new=score_response(model,s.prompt_token_ids,s.response_token_ids,grad_enabled=True)
                metrics=no_update_metrics(old,new,1e-5)
                reward=rw[s.group_id,s.sample_index]
                record={"group_id":s.group_id,"sample_index":s.sample_index,
                    "response_token_ids":s.response_token_ids,"sampler_behavior_token_log_probs":s.behavior_token_log_probs,
                    "trainer_old_token_log_probs":old.cpu().tolist(),"trainer_new_token_log_probs":new.detach().cpu().tolist(),
                    "no_update_check":metrics, **{k:reward[k] for k in ("eligible","reward","advantage")}}
                stream.write(json.dumps(record,allow_nan=False)+"\n"); stream.flush(); records.append(record)
                del old,new
    torch.cuda.synchronize()
    checks={"scoring_seconds":time.perf_counter()-started,"initial_parameter_sha256":initial,
        "final_parameter_sha256":parameter_digest(model),"no_grad_accumulated":all(p.grad is None for p in model.parameters()),
        "model_backward_calls":0,"optimizer_steps":0}
    write(output/"trainer_checks.json",checks)
    write(output/"reward_audit.json",rewards)
    torch.save(pack_trainer_scores(groups,records,tokenizer.pad_token_id),output/"trainer_score_batch.pt")
    print(json.dumps({"stage":"score",**checks}),flush=True)


def audit(output):
    import torch
    config, closeout, baseline, base, tokenizer, rows, groups, packed, rewards=load_run(output)
    records=[json.loads(x) for x in (output/"trainer_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    flat=[s for g in groups for s in g]; rw={(r["group_id"],r["sample_index"]):r for r in rewards["records"]}
    if len(records)!=192: raise ValueError("incomplete trainer scores")
    for sample,row in zip(flat,records):
        key=sample.group_id,sample.sample_index
        if ((row["group_id"],row["sample_index"])!=key or row["sampler_behavior_token_log_probs"]!=sample.behavior_token_log_probs
            or any(row[k]!=rw[key][k] for k in ("eligible","reward","advantage"))):
            raise ValueError("trainer/reward mapping differs")
        old=torch.tensor(row["trainer_old_token_log_probs"],dtype=torch.float32)
        new=torch.tensor(row["trainer_new_token_log_probs"],dtype=torch.float32,requires_grad=True)
        # This reproduces numeric fields only; runtime graph flags come from GPU stage.
        if no_update_metrics(old,new,1e-5)!=row["no_update_check"]: raise ValueError("old/new metrics differ")
    batch=pack_trainer_scores(groups,records,tokenizer.pad_token_id)
    saved=torch.load(output/"trainer_score_batch.pt",map_location="cpu",weights_only=True)
    if set(saved)!=set(batch) or any(not torch.equal(v,saved[k]) for k,v in batch.items()):
        raise ValueError("saved score tensors or masks differ")
    if read(output/"reward_audit.json")!=rewards: raise ValueError("saved reward differs")
    gen=read(output/"generation_checks.json"); checks=read(output/"trainer_checks.json")
    _, _, _, _, _, cases, prompts=setup()
    expected_effective={"dtype":"torch.float16", "logprobs_mode":"raw_logprobs", "max_model_len":512,
        "enforce_eager":True, "enable_prefix_caching":False, "enable_chunked_prefill":True,
        "max_num_seqs":4, "max_num_batched_tokens":512, "max_lora_rank":8}
    expected_sampling=dict(n=1,temperature=1,top_p=1,top_k=-1,max_tokens=64,min_tokens=0,
        repetition_penalty=1,presence_penalty=0,frequency_penalty=0,logprobs=1,
        stop_token_ids=[tokenizer.eos_token_id],ignore_eos=False,detokenize=False)
    if gen["effective_engine"]!=expected_effective or gen["sampling_parameters"]!=expected_sampling or gen["generation_requests"]!=224:
        raise ValueError("saved engine/sampling configuration differs")
    if not math.isfinite(gen["generation_seconds"]) or gen["generation_seconds"]<=0:
        raise ValueError("positive generation time required")
    for mode in ("benchmark_batch_samples","benchmark_serial_samples"):
        if len(gen[mode])!=16: raise ValueError("incomplete benchmark records")
        bench_rows=[]
        for j,sample in enumerate(gen[mode]):
            case=cases[j//4]
            reward=score_reward(sample["text"],case,finish_reason=sample["finish_reason"])
            bench_rows.append({"sample":sample,"case":asdict(case),"score":reward["score"],"semantic_pass":reward["semantic_pass"]})
        validate_records(bench_rows,cases[:4],prompts[:4],tokenizer,config)
    bench=benchmark_summary([RolloutSample(**s) for s in gen["benchmark_batch_samples"]],
        [RolloutSample(**s) for s in gen["benchmark_serial_samples"]],
        gen["benchmark"]["batch_seconds"],gen["benchmark"]["serial_seconds"])
    if bench!=gen["benchmark"]: raise ValueError("benchmark statistics differ")
    oldnew=all(r["no_update_check"]["passed"] and r["no_update_check"]["ratio_requires_grad"] for r in records)
    frozen=(checks["initial_parameter_sha256"]==checks["final_parameter_sha256"]==read(ROOT/"configs/trainer_probability_contract.json")["frozen_fp32_parameter_sha256"]
        and checks["no_grad_accumulated"] and checks["optimizer_steps"]==checks["model_backward_calls"]==0)
    adv=torch.tensor([[rw[g[0].group_id,i]["advantage"] for i in range(4)] for g in groups])
    correction=read(ROOT/"configs/rollout_correction_contract.json")
    loss,weight_stats=corrected_grpo_policy_loss(batch["trainer_new_log_probs"],batch["trainer_old_log_probs"],
        batch["sampler_behavior_log_probs"],adv,batch["eligible_response_mask"],
        upper_cap=correction["upper_cap"],clip_epsilon=correction["clip_epsilon"])
    result={"contract_version":closeout["contract_version"],"artifact_audit_passed":True,
        "phase4_cloud_run_completed":True,"same_path_no_update_ratio_passed":oldnew,"parameters_unchanged":frozen,
        "samples":192,"groups":48,"generation_requests":224,"benchmark":bench,
        "response_tokens":int(batch["response_mask"].sum()),"finish_reasons":dict(Counter(s.finish_reason for s in flat)),
        "strict_pass":sum(r["score"]["end_to_end_pass"] for r in rows),"reward_sum":rewards["reward_sum"],
        "semantic_pass":sum(r["semantic_pass"] for r in rows),
        "invalid_json":sum(not r["score"]["json_valid"] for r in rows),
        "mean_response_tokens":sum(len(s.response_token_ids) for s in flat)/len(flat),
        "max_old_new_logp_delta":max(r["no_update_check"]["max_abs_logp_delta"] for r in records),
        "max_ratio_minus_one":max(r["no_update_check"]["max_abs_ratio_minus_one"] for r in records),
        "sampler_trainer_max_logp_difference":max(abs(p-q) for r in records
            for p,q in zip(r["trainer_old_token_log_probs"],r["sampler_behavior_token_log_probs"])),
        "zero_variance_rate":rewards["zero_variance_rate"],"nonzero_variance_groups":48-rewards["zero_variance_groups"],
        "has_reward_learning_signal":rewards["has_reward_learning_signal"],"correction_weights":weight_stats,
        "corrected_policy_loss_diagnostic":loss.item(),"generation_seconds":gen["generation_seconds"],
        "generation_response_tokens_per_second":gen["generation_response_tokens_per_second"],
        "scoring_seconds":checks["scoring_seconds"],"original_cross_engine_gate_passed":False,
        "new_contract_interface_passed":bool(oldnew and frozen and gen["engine_shutdown_called"]),
        "training_admitted":False,"model_backward_calls":0,"optimizer_steps":0,
        "completion_review_pending":True,
        "files_sha256":{name:file_hash(output/name) for name in ("run_manifest.json","generation_checks.json",
            "rollouts.jsonl","trainer_scores.jsonl","trainer_checks.json","reward_audit.json","trainer_score_batch.pt")}}
    write(output/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("benchmark","files_sha256")}),flush=True)
    if not result["new_contract_interface_passed"]: raise SystemExit(2)


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    if (ROOT.parent/"phase4-closeout-review-20261021.zip").exists():
        raise FileExistsError("review archive already exists; preserve it before another run")
    price=args.hourly_price if args.hourly_price is not None else float(input("L40S hourly price (yuan): "))
    spent=args.spent_yuan if args.spent_yuan is not None else float(input("Total rental cost already spent (yuan): "))
    budget,seconds=budget_limits(price,spent)
    print(f"Hard run time limit: {seconds:.0f}s; estimated run-cost cap: {budget:.2f} yuan. Idle/storage costs excluded.",flush=True)
    started=time.perf_counter(); stages=[]; error=None
    try:
        for stage in ("generate","score","audit"):
            remaining=seconds-(time.perf_counter()-started)
            if remaining<=0: raise TimeoutError("closeout wall time exceeded")
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--stage",stage,"--output",str(args.output.resolve())],
                cwd=ROOT,start_new_session=(os.name!="nt"))
            try: code=proc.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name!="nt": os.killpg(proc.pid,signal.SIGTERM)
                else: proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name!="nt": os.killpg(proc.pid,signal.SIGKILL)
                    else: proc.kill()
                    proc.wait()
                raise TimeoutError("cloud stage stopped at bounded wall time")
            stages.append({"stage":stage,"exit_code":code})
            if code: raise RuntimeError(f"stage {stage} exited {code}; no automatic retry")
    except (Exception, KeyboardInterrupt) as exc: error=f"{type(exc).__name__}: {exc}"
    finally:
        if args.output.exists():
            elapsed=time.perf_counter()-started
            write(args.output/"cost_and_run.json",{"hourly_price_yuan":price,"previous_spend_yuan":spent,
                "elapsed_run_seconds":elapsed,"estimated_run_cost_yuan":elapsed*price/3600,
                "max_run_seconds":seconds,"max_run_budget_yuan":budget,"stages":stages,"error":error,
                "billing_scope":"job elapsed estimate only; idle, prior boot/download, storage, rounding excluded",
                "instance_automatically_shutdown":False})
            archive=ROOT.parent/"phase4-closeout-review-20261021.zip"
            if archive.exists(): raise FileExistsError(archive)
            with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
                for path in sorted(args.output.rglob("*")):
                    if path.is_file(): z.write(path,str(path.relative_to(args.output.parent)))
            print(f"Download review archive: {archive}",flush=True)
    if error: raise RuntimeError(error)
    print("Cloud run complete. Stop the instance when finished downloading; final Phase 4 review pending.",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("run","generate","score","audit"),default="run")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/trajectories/phase4-closeout-20261021-01")
    parser.add_argument("--hourly-price",type=float)
    parser.add_argument("--spent-yuan",type=float)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",VLLM_WORKER_MULTIPROC_METHOD="spawn",VLLM_NO_USAGE_STATS="1")
    if args.stage=="run": run(args)
    elif args.stage=="generate": generate(args.output)
    elif args.stage=="score": score(args.output)
    else: audit(args.output)


if __name__=="__main__": main()
