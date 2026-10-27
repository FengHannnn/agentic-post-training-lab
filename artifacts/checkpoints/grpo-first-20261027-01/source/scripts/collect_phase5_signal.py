"""Bounded 144x8 rollout and FP32 signal audit, with no model updates."""

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

from collect_vllm_rollout import ROOT, context, read, write, check_gpu, effective_config, inventory
from verify_cloud_bundle import file_hash
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.phase5_signal import (
    select_cases, sample_seed, sample_from_output, validate_records, signal_summary, run_seconds)
from agentic_post_training.rollouts.trainer_scores import score_response, no_update_metrics, pack_trainer_scores
from agentic_post_training.algorithms.rollout_correction import corrected_grpo_policy_loss

PLAN_PATH=ROOT/"configs/phase5_rollout_signal.json"
ARCHIVE=ROOT/"phase5-rollout-review-20261021.zip"
EFFECTIVE={"dtype":"torch.float16","logprobs_mode":"raw_logprobs","max_model_len":512,
    "enforce_eager":True,"enable_prefix_caching":False,"enable_chunked_prefill":True,
    "max_num_seqs":4,"max_num_batched_tokens":512,"max_lora_rank":8}


def sampling(eos):
    return dict(n=1,temperature=1,top_p=1,top_k=-1,max_tokens=64,min_tokens=0,
        repetition_penalty=1,presence_penalty=0,frequency_penalty=0,logprobs=1,
        stop_token_ids=[eos],ignore_eos=False,detokenize=False)


def setup():
    config,baseline,base,tokenizer,_,_=context()
    plan=read(PLAN_PATH)
    if (plan["contract_version"]!="phase5-signal-144x8-v1" or plan["group_size"]!=8
        or plan["base_seed"]!=20261021 or plan["seed_group_stride"]!=1000 or plan["chunk_requests"]!=64
        or plan["max_wall_seconds"]!=1800 or plan["max_run_cost_yuan"]!=10
        or plan["old_new_logp_tolerance"]!=1e-5 or plan["training_admitted"] or plan["optimizer_steps"]!=0):
        raise ValueError("fixed Phase 5 signal contract differs")
    prior=ROOT/"artifacts/trajectories/phase4-closeout-20261021-01/summary.json"
    if file_hash(prior)!=plan["prior_phase4_summary_sha256"]:
        raise ValueError("bound completed Phase 4 result required")
    gate=read(prior)
    if not gate["new_contract_interface_passed"] or gate["training_admitted"]:
        raise ValueError("completed no-update Phase 4 interface required")
    folder=ROOT/"data/preferences/dpo_v2"
    cases=select_cases(load_preference_splits(folder,read(folder/"manifest.json"))["train"],plan["case_ids"])
    prompts=[tokenizer.apply_chat_template([{"role":"system","content":c.system},
        {"role":"user","content":c.prompt}],tokenize=True,add_generation_prompt=True) for c in cases]
    if any(len(p)+64>512 for p in prompts): raise ValueError("context budget exceeded")
    return config,plan,baseline,base,tokenizer,cases,prompts


def identity(config,plan,baseline):
    inv=inventory(config,baseline)
    inv["phase5_plan"]=plan
    inv["source_sha256"]={k.replace('\\','/'):v for k,v in inv["source_sha256"].items()}
    files=[Path(__file__),PLAN_PATH,ROOT/"src/agentic_post_training/rollouts/phase5_signal.py",
        ROOT/"src/agentic_post_training/rollouts/trainer_scores.py",
        ROOT/"src/agentic_post_training/algorithms/rollout_correction.py",
        ROOT/"src/agentic_post_training/trainers/dpo.py",
        ROOT/"configs/trainer_probability_contract.json",ROOT/"configs/rollout_correction_contract.json"]
    inv["source_sha256"].update({str(p.relative_to(ROOT)).replace('\\','/'):file_hash(p) for p in files})
    return inv


def load_run(output):
    config,plan,baseline,base,tokenizer,cases,prompts=setup()
    saved=read(output/"run_manifest.json"); expected=identity(config,plan,baseline)
    if any(saved[k]!=v for k,v in expected.items()): raise ValueError("model/data/source identity differs")
    for rel,digest in saved["source_sha256"].items():
        if file_hash(output/"source"/rel)!=digest: raise ValueError("source snapshot differs")
    rows=[json.loads(x) for x in (output/"rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    groups,packed,rewards=validate_records(rows,cases,prompts,tokenizer)
    return config,plan,baseline,base,tokenizer,rows,groups,packed,rewards


def generate(output):
    config,plan,baseline,base,tok,cases,prompts=setup()
    device=check_gpu(config)
    output.mkdir(parents=True,exist_ok=False)
    inv=identity(config,plan,baseline);inv["device"]=device
    write(output/"run_manifest.json",inv)
    for rel in inv["source_sha256"]:
        dest=output/"source"/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,dest)
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch
    import torch.distributed as dist
    started=time.perf_counter()
    engine=LLM(model=str(base),tokenizer=str(baseline/"checkpoint/tokenizer"),dtype="half",
        trust_remote_code=False,enable_lora=True,max_lora_rank=8,max_loras=1,max_model_len=512,
        max_num_seqs=4,max_num_batched_tokens=512,gpu_memory_utilization=.25,enforce_eager=True,
        enable_prefix_caching=False,enable_chunked_prefill=True,seed=plan["base_seed"],
        generation_config="vllm",logprobs_mode="raw_logprobs")
    load_seconds=time.perf_counter()-started
    effective=effective_config(engine)
    shared=sampling(tok.eos_token_id)
    entries=[(g,i,sample_seed(g,i)) for g in range(144) for i in range(8)]
    request=LoRARequest("sft-v2-phase3",1,str(baseline/"checkpoint/adapter"))
    try:
        if effective!=EFFECTIVE: raise ValueError("effective native engine settings differ")
        torch.cuda.synchronize();started=time.perf_counter();tokens=0
        with (output/"rollouts.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
            for start in range(0,len(entries),64):
                chunk=entries[start:start+64]
                params=[SamplingParams(seed=seed,**shared) for g,i,seed in chunk]
                results=engine.generate([{"prompt_token_ids":prompts[g]} for g,i,seed in chunk],
                    params,lora_request=request,use_tqdm=False)
                if len(results)!=len(chunk): raise ValueError("missing native outputs")
                for result,(g,i,seed) in zip(results,chunk):
                    s=sample_from_output(result,prompts[g],cases[g],i,seed,tok)
                    rw=score_reward(s.text,cases[g],finish_reason=s.finish_reason)
                    stream.write(json.dumps({"sample":asdict(s),"case":asdict(cases[g]),
                        "score":rw["score"],"semantic_pass":rw["semantic_pass"]},ensure_ascii=False,allow_nan=False)+"\n")
                    tokens+=len(s.response_token_ids)
                stream.flush()
                print(f"Saved rollout {start+len(chunk)}/1152",flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
    finally:
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized(): dist.destroy_process_group()
    write(output/"generation_checks.json",{"effective_engine":effective,"sampling_parameters":shared,
        "engine_load_seconds":load_seconds,"generation_seconds":seconds,
        "response_tokens":tokens,"generation_response_tokens_per_second":tokens/seconds,
        "generation_requests":1152,"engine_generate_calls":18,"chunk_requests":64,"engine_shutdown_called":True})


def score(output):
    if any((output/p).exists() for p in ("trainer_scores.jsonl","trainer_checks.json","reward_audit.json","trainer_score_batch.pt")):
        raise FileExistsError("score output exists; refusing to overwrite")
    config,plan,baseline,base,tok,rows,groups,packed,rewards=load_run(output)
    if check_gpu(config)!=read(output/"run_manifest.json")["device"]: raise ValueError("GPU model changed")
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from agentic_post_training.trainers.dpo import parameter_digest
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation="eager")
    model=PeftModel.from_pretrained(model,str(baseline/"checkpoint/adapter"),is_trainable=True,
        autocast_adapter_dtype=False).to("cuda").eval()
    before=parameter_digest(model)
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if (before!=read(ROOT/"configs/trainer_probability_contract.json")["frozen_fp32_parameter_sha256"]
        or sum(p.numel() for n,p in trainable)!=230400 or any("lora_" not in n for n,p in trainable)
        or any(p.dtype!=torch.float32 for p in model.parameters())
        or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules())):
        raise ValueError("original FP32 SFT/LoRA identity differs")
    rw={(r["group_id"],r["sample_index"]):r for r in rewards["records"]}
    records=[];torch.cuda.synchronize();started=time.perf_counter()
    with (output/"trainer_scores.jsonl").open("x",encoding="utf-8",newline="\n") as stream:
        for g,group in enumerate(groups):
            for s in group:
                old=score_response(model,s.prompt_token_ids,s.response_token_ids,grad_enabled=False)
                new=score_response(model,s.prompt_token_ids,s.response_token_ids,grad_enabled=True)
                metrics=no_update_metrics(old,new,1e-5)
                reward=rw[s.group_id,s.sample_index]
                row={"group_id":s.group_id,"sample_index":s.sample_index,"response_token_ids":s.response_token_ids,
                    "sampler_behavior_token_log_probs":s.behavior_token_log_probs,
                    "trainer_old_token_log_probs":old.cpu().tolist(),"trainer_new_token_log_probs":new.detach().cpu().tolist(),
                    "no_update_check":metrics,**{k:reward[k] for k in ("eligible","reward","advantage")}}
                stream.write(json.dumps(row,allow_nan=False)+"\n");records.append(row)
                del old,new
            stream.flush()
            if (g+1)%12==0: print(f"Scored groups {g+1}/144",flush=True)
    torch.cuda.synchronize()
    write(output/"trainer_checks.json",{"scoring_seconds":time.perf_counter()-started,
        "initial_parameter_sha256":before,"final_parameter_sha256":parameter_digest(model),
        "no_grad_accumulated":all(p.grad is None for p in model.parameters()),"model_backward_calls":0,"optimizer_steps":0})
    write(output/"reward_audit.json",rewards)
    torch.save(pack_trainer_scores(groups,records,tok.pad_token_id),output/"trainer_score_batch.pt")


def audit(output):
    import torch
    config,plan,baseline,base,tok,rows,groups,packed,rewards=load_run(output)
    records=[json.loads(x) for x in (output/"trainer_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    flat=[s for g in groups for s in g]
    rw={(r["group_id"],r["sample_index"]):r for r in rewards["records"]}
    if len(records)!=1152: raise ValueError("incomplete trainer records")
    for s,row in zip(flat,records):
        key=s.group_id,s.sample_index
        if ((row["group_id"],row["sample_index"])!=key or row["sampler_behavior_token_log_probs"]!=s.behavior_token_log_probs
            or any(row[k]!=rw[key][k] for k in ("eligible","reward","advantage"))):
            raise ValueError("trainer scores/reward mapping differs")
        old=torch.tensor(row["trainer_old_token_log_probs"],dtype=torch.float32)
        new=torch.tensor(row["trainer_new_token_log_probs"],dtype=torch.float32,requires_grad=True)
        if no_update_metrics(old,new,1e-5)!=row["no_update_check"]: raise ValueError("saved no-update statistics differ")
    batch=pack_trainer_scores(groups,records,tok.pad_token_id)
    saved=torch.load(output/"trainer_score_batch.pt",map_location="cpu",weights_only=True)
    if set(saved)!=set(batch) or any(not torch.equal(v,saved[k]) for k,v in batch.items()):
        raise ValueError("saved scores/masks differ")
    if read(output/"reward_audit.json")!=rewards: raise ValueError("saved reward differs")
    gen=read(output/"generation_checks.json");checks=read(output/"trainer_checks.json")
    if (gen["effective_engine"]!=EFFECTIVE or gen["sampling_parameters"]!=sampling(tok.eos_token_id)
        or gen["generation_requests"]!=1152 or gen["engine_generate_calls"]!=18 or gen["chunk_requests"]!=64
        or not math.isfinite(gen["generation_seconds"]) or gen["generation_seconds"]<=0
        or gen["response_tokens"]!=int(batch["response_mask"].sum())):
        raise ValueError("saved generation configuration/timing differs")
    expected=read(ROOT/"configs/trainer_probability_contract.json")["frozen_fp32_parameter_sha256"]
    frozen=(checks["initial_parameter_sha256"]==checks["final_parameter_sha256"]==expected
        and checks["no_grad_accumulated"] and checks["model_backward_calls"]==checks["optimizer_steps"]==0)
    oldnew=all(r["no_update_check"]["passed"] and r["no_update_check"]["ratio_requires_grad"] for r in records)
    adv=torch.tensor([[rw[g[0].group_id,i]["advantage"] for i in range(8)] for g in groups])
    correction=read(ROOT/"configs/rollout_correction_contract.json")
    loss,stats=corrected_grpo_policy_loss(batch["trainer_new_log_probs"],batch["trainer_old_log_probs"],
        batch["sampler_behavior_log_probs"],adv,batch["eligible_response_mask"],
        upper_cap=correction["upper_cap"],clip_epsilon=correction["clip_epsilon"])
    signals=signal_summary(rewards)
    result={"contract_version":plan["contract_version"],"artifact_audit_passed":True,"samples":1152,"groups":144,
        "generation_requests":1152,"response_tokens":int(batch["response_mask"].sum()),
        "finish_reasons":dict(Counter(s.finish_reason for s in flat)),"strict_pass":sum(r["score"]["end_to_end_pass"] for r in rows),
        "semantic_pass":sum(r["semantic_pass"] for r in rows),"invalid_json":sum(not r["score"]["json_valid"] for r in rows),
        "reward_sum":rewards["reward_sum"],"zero_variance_rate":rewards["zero_variance_rate"],
        "has_reward_learning_signal":rewards["has_reward_learning_signal"],**signals,
        "same_path_no_update_ratio_passed":oldnew,"parameters_unchanged":frozen,
        "max_old_new_logp_delta":max(r["no_update_check"]["max_abs_logp_delta"] for r in records),
        "max_ratio_minus_one":max(r["no_update_check"]["max_abs_ratio_minus_one"] for r in records),
        "sampler_trainer_max_logp_difference":max(abs(p-q) for r in records
            for p,q in zip(r["trainer_old_token_log_probs"],r["sampler_behavior_token_log_probs"])),
        "correction_weights":stats,"corrected_policy_loss_diagnostic":loss.item(),
        "generation_seconds":gen["generation_seconds"],
        "generation_response_tokens_per_second":gen["response_tokens"]/gen["generation_seconds"],
        "scoring_seconds":checks["scoring_seconds"],"original_cross_engine_gate_passed":False,
        "signal_collection_interface_passed":bool(oldnew and frozen and gen["engine_shutdown_called"]),
        "training_admitted":False,"model_backward_calls":0,"optimizer_steps":0,
        "files_sha256":{name:file_hash(output/name) for name in ("run_manifest.json","generation_checks.json",
            "rollouts.jsonl","trainer_scores.jsonl","trainer_checks.json","reward_audit.json","trainer_score_batch.pt")}}
    write(output/"summary.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="files_sha256"},ensure_ascii=False),flush=True)
    if not result["signal_collection_interface_passed"]: raise SystemExit(2)


def run(args):
    if args.output.exists() or ARCHIVE.exists(): raise FileExistsError("output or archive exists; no automatic retry")
    price=args.hourly_price if args.hourly_price is not None else float(input("GPU hourly price (yuan): "))
    seconds=run_seconds(price)
    print(f"At most {seconds:.0f}s, estimated job cost <= 10 yuan; no global cumulative budget prompt.",flush=True)
    started=time.perf_counter();stages=[];error=None
    try:
        for stage in ("generate","score","audit"):
            remaining=seconds-(time.perf_counter()-started)
            if remaining<=0: raise TimeoutError("bounded job duration exceeded")
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"--stage",stage,"--output",str(args.output.resolve())],
                cwd=ROOT,start_new_session=(os.name!="nt"))
            try: code=proc.wait(timeout=remaining)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                if os.name!="nt": os.killpg(proc.pid,signal.SIGTERM)
                else: proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name!="nt": os.killpg(proc.pid,signal.SIGKILL)
                    else: proc.kill()
                    proc.wait()
                raise TimeoutError("stage stopped; no retry")
            stages.append({"stage":stage,"exit_code":code})
            if code: raise RuntimeError(f"stage {stage} exited {code}")
    except (Exception,KeyboardInterrupt) as exc: error=f"{type(exc).__name__}: {exc}"
    finally:
        if args.output.exists():
            elapsed=time.perf_counter()-started
            write(args.output/"cost_and_run.json",{"hourly_price_yuan":price,"elapsed_run_seconds":elapsed,
                "estimated_run_cost_yuan":elapsed*price/3600,"max_run_seconds":seconds,"max_run_cost_yuan":10,
                "stages":stages,"error":error,"global_200_yuan_limit_enforced":False,
                "billing_scope":"job elapsed estimate only; boot/idle/storage/rounding excluded",
                "instance_automatically_shutdown":False})
            with zipfile.ZipFile(ARCHIVE,"x",zipfile.ZIP_DEFLATED) as z:
                for p in sorted(args.output.rglob("*")):
                    if p.is_file(): z.write(p,str(p.relative_to(args.output.parent)))
            print(f"Download from project root: {ARCHIVE}",flush=True)
    if error: raise RuntimeError(error)
    print("Signal collection finished; no model update. Download archive and stop idle GPU.",flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("run","generate","score","audit"),default="run")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/trajectories/phase5-signal-20261021-01")
    parser.add_argument("--hourly-price",type=float)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",VLLM_WORKER_MULTIPROC_METHOD="spawn",VLLM_NO_USAGE_STATS="1")
    if args.stage=="run":run(args)
    elif args.stage=="generate":generate(args.output)
    elif args.stage=="score":score(args.output)
    else:audit(args.output)


if __name__=="__main__":main()
