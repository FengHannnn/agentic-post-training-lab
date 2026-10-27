"""Compare serial and parallel GRPO execution; discard every probe update."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile

import train_grpo_first as first
from train_grpo_first import ROOT,read,write,file_hash
from agentic_post_training.rollouts.trainer_scores import score_response
from agentic_post_training.rollouts.phase5_signal import run_seconds
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (
    accumulate_minibatch,guarded_step,lora_parameters,restore_optimizer_checkpoint,
    adapter_snapshot,compare_adapter_snapshot,optimizer_snapshot,compare_optimizer_snapshot)
from agentic_post_training.trainers.grpo_microbatch import accumulate_microbatches,guarded_microbatch_step

PLAN=ROOT/'configs/grpo_microbatch_benchmark.json'
ARCHIVE=ROOT/'grpo-microbatch-review-20261027.zip'


def load_inputs():
    plan=read(PLAN)
    fixed={'contract_version':'grpo-microbatch-benchmark-v1','microbatch_sizes':[1,4,8],
        'effective_minibatch_groups':4,'effective_completions':32,'timing_repeats':3,
        'logp_absolute_tolerance':1e-5,'loss_absolute_tolerance':1e-6,
        'gradient_max_absolute_tolerance':1e-6,'gradient_relative_l2_tolerance':1e-4,
        'adapter_absolute_tolerance':1e-6,'optimizer_absolute_tolerance':1e-7,
        'published_candidate_updates':0,'max_wall_seconds':1800,'max_run_cost_yuan':10.}
    if any(plan[k]!=v for k,v in fixed.items()):raise ValueError('fixed benchmark contract differs')
    source=ROOT/plan['source_training_run'];summary=read(source/'summary.json')
    if (file_hash(source/'summary.json')!=plan['source_training_summary_sha256']
        or not summary['engineering_checks_passed'] or summary['logical_optimizer_steps']!=16):
        raise ValueError('audited first-update checkpoint required')
    for rel,digest in summary['files_sha256'].items():
        if file_hash(source/rel)!=digest:raise ValueError('first-update artifact changed')
    manifest=read(source/'run_manifest.json')
    for rel,digest in manifest['source_sha256'].items():
        if file_hash(ROOT/rel)!=digest:raise ValueError('first-update source changed')
    training,_,cfg,baseline,base,tok,groups,targets,_=first.context()
    indexed={g[0].group_id:g for g in groups}
    group_ids=training['group_batches'][0]
    samples=[s for key in group_ids for s in indexed[key]]
    if len(group_ids)!=4 or len(samples)!=32:raise ValueError('same 4 groups / 32 answers required')
    return plan,source,summary,manifest,training,cfg,baseline,base,tok,samples,targets


def gradient_difference(actual,expected):
    if set(actual)!=set(expected) or not actual:raise ValueError('gradient names differ')
    squared_error=squared_reference=0.;maximum=0.
    for key,value in actual.items():
        if value.shape!=expected[key].shape:raise ValueError('gradient shapes differ')
        delta=value.double()-expected[key].double()
        if not delta.isfinite().all() or not expected[key].isfinite().all():raise ValueError('nonfinite gradient')
        maximum=max(maximum,delta.abs().max().item())
        squared_error+=delta.square().sum().item()
        squared_reference+=expected[key].double().square().sum().item()
    relative=math.sqrt(squared_error/squared_reference) if squared_reference else (0. if not squared_error else float('inf'))
    return {'gradient_max_abs_error':maximum,'gradient_relative_l2_error':relative}


def numerical_gate(metrics,plan):
    names={'logp_max_abs_error':'logp_absolute_tolerance','loss_abs_error':'loss_absolute_tolerance',
        'gradient_max_abs_error':'gradient_max_absolute_tolerance',
        'gradient_relative_l2_error':'gradient_relative_l2_tolerance',
        'max_adapter_abs_error':'adapter_absolute_tolerance','max_optimizer_state_abs_error':'optimizer_absolute_tolerance'}
    return all(math.isfinite(metrics[k]) and metrics[k]<=plan[v] for k,v in names.items())


def recommend(checks,timing):
    medians={size:statistics.median(r['seconds'] for r in timing if r['microbatch_size']==size)
        for size in (1,4,8)}
    if any(not math.isfinite(t) or t<=0 for t in medians.values()):raise ValueError('positive finite timing required')
    allowed=[size for size in (4,8) if checks[str(size)]['passed'] and medians[size]<medians[1]]
    selected=min(allowed,key=lambda size:medians[size]) if allowed else 1
    return {'recommended_microbatch_size':selected,'median_seconds':{str(k):v for k,v in medians.items()},
        'median_speedup_over_serial':{str(k):medians[1]/v for k,v in medians.items()},
        'selection_rule':'fastest numerically passing parallel size faster than serial; otherwise retain serial',
        'scope':'one fixed 32-answer minibatch on this GPU; no general throughput claim'}


def benchmark(output):
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    plan,source,summary,manifest,training,cfg,baseline,base,tok,samples,targets=load_inputs()
    output.mkdir(parents=True,exist_ok=False)
    sources=manifest['source_sha256'].copy()
    for rel in ('scripts/benchmark_grpo_microbatch.py','configs/grpo_microbatch_benchmark.json',
                'src/agentic_post_training/rollouts/batched_scores.py',
                'src/agentic_post_training/trainers/grpo_microbatch.py'):
        sources[rel]=file_hash(ROOT/rel)
    write(output/'run_manifest.json',{'plan':plan,'device':first.check_gpu(cfg),'source_sha256':sources,
        'source_training_summary_sha256':file_hash(source/'summary.json'),
        'group_ids':list(dict.fromkeys(s.group_id for s in samples)),
        'sample_keys':[[s.group_id,s.sample_index] for s in samples],
        'starting_logical_step':16,'temporary_probe_updates_are_discarded':True,
        'correctness_probes_also_warm_up_each_mode':True,'timing_excludes_restore_and_cpu_snapshots':True})
    for rel,digest in sources.items():
        dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        if file_hash(dest)!=digest:raise ValueError('source snapshot differs')
    checkpoint=source/'final-checkpoint'
    policy=first.load_model(base,checkpoint/'adapter',True)
    reference=first.load_model(base,baseline/'checkpoint/adapter',False)
    initial=parameter_digest(policy);frozen=parameter_digest(policy,frozen_only=True);ref_digest=parameter_digest(reference)
    if initial!=summary['final_parameter_sha256'] or ref_digest!=summary['initial_parameter_sha256']:
        raise ValueError('candidate/reference identity differs')
    initial_adapter=adapter_snapshot(policy);cache={};reference_error=0.
    for s in samples:
        key=s.group_id,s.sample_index
        cache[key]=score_response(reference,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu()
        expected=torch.tensor(targets[key]['trainer_old_token_log_probs'],dtype=torch.float32)
        reference_error=max(reference_error,(cache[key]-expected).abs().max().item())
    if reference_error>1e-5:raise ValueError('reference/stored old scores differ')

    def reset():
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name,p in policy.named_parameters():
                if p.requires_grad:p.copy_(initial_adapter[name].to(p.device))
        optimizer=torch.optim.AdamW(lora_parameters(policy),lr=training['learning_rate'],weight_decay=0.)
        if restore_optimizer_checkpoint(policy,optimizer,checkpoint,manifest['checkpoint_input_digest'])!=16:
            raise ValueError('optimizer must restart at step 16')
        policy.eval()
        return optimizer

    kwargs={k:training[k] for k in ('beta','clip_epsilon','tis_cap')}
    checks={};baseline_scores=None;baseline_gradients=None;baseline_adapter=None;baseline_optimizer=None;baseline_loss=None
    for size in plan['microbatch_sizes']:
        optimizer=reset()
        if size==1:
            scores=[score_response(policy,s.prompt_token_ids,s.response_token_ids,grad_enabled=False).cpu().tolist() for s in samples]
            metrics=accumulate_minibatch(policy,samples,targets,cache,**kwargs)
        else:
            metrics=accumulate_microbatches(policy,samples,targets,cache,microbatch_size=size,
                pad_token_id=tok.pad_token_id,capture_scores=True,**kwargs)
            scores=metrics.pop('scored_new_log_probs')
        gradients={n:p.grad.detach().cpu().clone() for n,p in policy.named_parameters() if p.requires_grad}
        if metrics['sampled_k3']>training['stop_sampled_k3'] or metrics['ratio_clip_fraction']>training['stop_ratio_clip_fraction']:
            raise ValueError('probe violates existing update stop guard')
        norm=torch.nn.utils.clip_grad_norm_(lora_parameters(policy),training['max_grad_norm'],error_if_nonfinite=True)
        if not torch.isfinite(norm) or norm.item()<=0:raise ValueError('nonzero finite gradient required')
        optimizer.step()
        if any(v['step'].item()!=17 for v in optimizer.state.values()):raise ValueError('temporary step counter differs')
        if size==1:
            baseline_scores=scores;baseline_gradients=gradients;baseline_loss=metrics['loss']
            baseline_adapter=adapter_snapshot(policy);baseline_optimizer=optimizer_snapshot(optimizer)
            difference={'logp_max_abs_error':0.,'loss_abs_error':0.,'gradient_max_abs_error':0.,
                'gradient_relative_l2_error':0.,'max_adapter_abs_error':0.,'max_optimizer_state_abs_error':0.}
        else:
            if len(scores)!=len(baseline_scores) or any(len(a)!=len(b) for a,b in zip(scores,baseline_scores)):
                raise ValueError('response score alignment differs')
            difference={'logp_max_abs_error':max(abs(a-b) for left,right in zip(scores,baseline_scores) for a,b in zip(left,right)),
                'loss_abs_error':abs(metrics['loss']-baseline_loss),**gradient_difference(gradients,baseline_gradients),
                **compare_adapter_snapshot(policy,baseline_adapter,plan['adapter_absolute_tolerance']),
                **compare_optimizer_snapshot(optimizer,baseline_optimizer,plan['optimizer_absolute_tolerance'])}
        checks[str(size)]={**difference,'passed':numerical_gate(difference,plan),'metrics':metrics,
            'grad_norm_before_clip':norm.item(),'temporary_optimizer_step':17}
        write(output/f'correctness-size{size}.json',checks[str(size)])
        print(json.dumps({'microbatch_size':size,'numeric_check':checks[str(size)]['passed'],**difference}),flush=True)
        del optimizer,gradients

    timing=[]
    for repeat,order in enumerate(((1,4,8),(4,8,1),(8,1,4)),1):
        for size in order:
            optimizer=reset();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            if size==1:metrics=guarded_step(policy,optimizer,samples,targets,cache,training)
            else:metrics=guarded_microbatch_step(policy,optimizer,samples,targets,cache,training,
                microbatch_size=size,pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            if not metrics['updated']:raise ValueError('timed probe stopped')
            row={'repeat':repeat,'microbatch_size':size,'seconds':elapsed,
                'response_tokens_per_second':metrics['response_tokens']/elapsed,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'forward_backward_calls':math.ceil(len(samples)/size),'metrics':metrics}
            timing.append(row);write(output/f'timing-repeat{repeat}-size{size}.json',row)
            print(json.dumps({k:v for k,v in row.items() if k!='metrics'}),flush=True)
            del optimizer
    write(output/'correctness.json',checks);write(output/'timing.json',timing)
    optimizer=reset();del optimizer
    restored=parameter_digest(policy)==initial
    frozen_intact=parameter_digest(policy,frozen_only=True)==frozen
    ref_intact=parameter_digest(reference)==ref_digest and all(p.grad is None and not p.requires_grad for p in reference.parameters())
    for rel,digest in summary['files_sha256'].items():
        if file_hash(source/rel)!=digest:raise ValueError('input artifact changed during benchmark')
    for rel,digest in sources.items():
        if file_hash(ROOT/rel)!=digest:raise ValueError('source changed during benchmark')
    intact=restored and frozen_intact and ref_intact
    result={'contract_version':plan['contract_version'],**recommend(checks,timing),'correctness':checks,
        'effective_groups':4,'effective_completions':32,'response_tokens':sum(len(s.response_token_ids) for s in samples),
        'temporary_optimizer_steps_discarded':12,'published_candidate_updates':0,'new_generation_calls':0,
        'candidate_restored_to_input_checkpoint':restored,'frozen_parameters_unchanged':frozen_intact,
        'reference_unchanged_and_no_grad':ref_intact,'input_artifacts_unchanged':True,
        'source_parameter_sha256':initial,'reference_old_max_logp_error':reference_error,
        'benchmark_completed':bool(intact),'baseline_promoted':False,
        'files_sha256':{str(p.relative_to(output)).replace('\\','/'):file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write(output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files_sha256','correctness')}),flush=True)
    if not intact:raise RuntimeError('benchmark state preservation failed')


def run(args):
    if args.output.exists() or ARCHIVE.exists():raise FileExistsError('fresh output/archive required; no automatic retry')
    price=args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
    seconds=run_seconds(price);started=time.perf_counter();error=None
    try:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--stage','benchmark','--output',str(args.output.resolve())],
            cwd=ROOT,start_new_session=(os.name!='nt'))
        try:code=proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
            else:proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                else:proc.kill()
                proc.wait()
            raise TimeoutError('bounded benchmark stopped; no automatic retry')
        if code:raise RuntimeError(f'benchmark exited {code}')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        if args.output.exists():
            elapsed=time.perf_counter()-started
            write(args.output/'cost_and_run.json',{'elapsed_run_seconds':elapsed,'hourly_price_yuan':price,
                'estimated_run_cost_yuan':elapsed*price/3600,'error':error,'max_run_seconds':seconds,
                'billing_scope':'job only; excludes boot/idle/storage/rounding','instance_automatically_shutdown':False})
            with zipfile.ZipFile(ARCHIVE,'x',zipfile.ZIP_DEFLATED) as z:
                for p in sorted(args.output.rglob('*')):
                    if p.is_file():z.write(p,str(p.relative_to(args.output.parent)))
            print(f'Download from project root: {ARCHIVE}',flush=True)
    if error:raise RuntimeError(error)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('run','benchmark'),default='run')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluations/grpo-microbatch-20261027-01')
    parser.add_argument('--hourly-price',type=float)
    args=parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.stage=='run':run(args)
    else:benchmark(args.output)


if __name__=='__main__':main()
