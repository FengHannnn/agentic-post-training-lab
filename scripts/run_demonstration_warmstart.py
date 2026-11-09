"""Bounded demonstration warm start, fresh sampling, GRPO and attribution audit."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

from demonstration_warmstart_common import ROOT, OUTPUT, PLAN, context, identity
from reward_comparison_common import import_comparison_adam
from collect_vllm_rollout import read, write
from agentic_post_training.rollouts.phase5_signal import run_seconds
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files, digest_json

ARCHIVE = ROOT / 'grpo-demonstration-warmstart-review-20261028.zip'
BUNDLE = ROOT / 'grpo_demonstration_warmstart_bundle_manifest.json'


def preflight(check_weights=False):
    ctx = context()
    plan = ctx['plan']
    inv = identity(ctx)
    result = {'contract_version':plan['contract_version'],'input_preflight_passed':True,
        'cases':24,'correct_demonstrations':len(ctx['encoded']),'logical_sft_steps':18,
        'planned_fresh_samples':768,'max_grpo_steps':6,'parent_adam_step':31,
        'source_files':len(inv['source_sha256']),'source_sha256':inv['source_sha256'],
        'gpu_runtime_checked':False,'model_forward_calls':0,'optimizer_updates':0,
        'cpu_weight_identity_checked':False,'plan_sha256':file_hash(PLAN),
        'demonstration_response_tokens':[sum(row['loss_mask']) for row in ctx['encoded']]}
    if check_weights:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from agentic_post_training.trainers.dpo import parameter_digest
        from agentic_post_training.trainers.grpo import lora_parameters
        torch.manual_seed(plan['seed'])
        policy = AutoModelForCausalLM.from_pretrained(str(ctx['base']), local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation='eager')
        policy = PeftModel.from_pretrained(policy, str(ctx['parent'] / 'final-checkpoint/adapter'), is_trainable=True,
                                           autocast_adapter_dtype=False).eval()
        actual = parameter_digest(policy)
        if actual != plan['parent_parameter_sha256']:
            raise ValueError('CPU parent reconstruction differs')
        optimizer = torch.optim.AdamW(lora_parameters(policy), lr=plan['learning_rate'], weight_decay=0.)
        bridge = import_comparison_adam(policy, optimizer, ctx['parent'] / 'final-checkpoint/optimizer.pt',
            expected_hash=plan['parent_optimizer_sha256'], old_binding=read(ctx['parent'] / 'run_manifest.json')['checkpoint_input_digest'],
            new_binding=digest_json({'preflight': file_hash(PLAN)}))
        if any(p.grad is not None for p in policy.parameters()):
            raise ValueError('unexpected preflight gradient')
        result.update(cpu_weight_identity_checked=True, reconstructed_parameter_sha256=actual, adam_import=bridge,
            trainable_parameters=sum(p.numel() for p in lora_parameters(policy)), trainable_tensors=sum(p.requires_grad for p in policy.parameters()))
    return result


def dispatch(price):
    seconds = run_seconds(price)
    bundle = read(BUNDLE)
    verify_files(ROOT, bundle['files'])
    verify_files(ROOT, bundle['required_existing_files'])
    if OUTPUT.exists() or ARCHIVE.exists():
        raise FileExistsError('fresh comparison output/archive required; no automatic retry')
    result = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / 'preflight.json', result)
    started = time.perf_counter()
    stages, error = [], None
    try:
        for stage in ('warmstart', 'generate', 'grpo', 'audit'):
            remaining = seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('comparison budget expired')
            proc = subprocess.Popen([sys.executable, str(ROOT / 'scripts/train_demonstration_warmstart.py'),
                '--stage', stage, '--output', str(OUTPUT)], cwd=ROOT, start_new_session=(os.name != 'nt'))
            try:
                code = proc.wait(timeout=remaining)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if os.name != 'nt': os.killpg(proc.pid, signal.SIGTERM)
                else: proc.terminate()
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name != 'nt': os.killpg(proc.pid, signal.SIGKILL)
                    else: proc.kill()
                    proc.wait()
                raise TimeoutError('comparison stopped; no automatic retry')
            stages.append({'stage': stage, 'exit_code': code})
            if code:
                raise RuntimeError(f'{stage} failed with exit code {code}; subsequent stages not started')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        elapsed = time.perf_counter() - started
        write(OUTPUT / 'cost_and_run.json', {'elapsed_run_seconds': elapsed, 'hourly_price_yuan': price,
            'estimated_run_cost_yuan': elapsed * price / 3600, 'max_run_seconds': seconds, 'max_run_cost_yuan': 10.,
            'stages': stages, 'error': error, 'automatic_retry': False, 'instance_automatically_shutdown': False,
            'billing_scope': 'all four bounded stages combined; excludes preflight/packing/boot/idle/storage/rounding'})
        with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUTPUT.rglob('*')):
                if path.is_file(): z.write(path, path.relative_to(OUTPUT.parent).as_posix())
        print(f'Download from project root: {ARCHIVE}', flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hourly-price', type=float)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--check-cpu-weights', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    if args.preflight:
        result = preflight(args.check_cpu_weights)
        if args.preflight_output: write(args.preflight_output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        price = args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
        dispatch(price)
