"""Sequential bounded refresh/update dispatch; stop after any failed phase."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from iteration_common import ROOT, PLAN_PATH, setup, identity, evaluation_suites, read
from collect_vllm_rollout import write
from agentic_post_training.trainers.policy_iteration import (
    verify_files, verify_parent, digest_json, file_hash, import_parent_adam, seed_for)

BUNDLE_MANIFEST = ROOT / 'grpo_iteration_bundle_manifest.json'
OUTPUT = ROOT / 'artifacts/checkpoints/grpo-iteration-20261028-01'
DRIVER = ROOT / 'artifacts/evaluations/grpo-iteration-driver-20261028-01'
ARCHIVE = ROOT / 'grpo-policy-iteration-review-20261028.zip'


def preflight(*, check_weights=False):
    """No GPU checks, sampling, forward/backward, or optimizer updates."""
    config, plan, baseline, base, tok, cases, prompts = setup()
    parent, summary, manifest = verify_parent(ROOT, plan)
    suites = evaluation_suites(plan)
    source_identity = identity(config, plan, baseline)
    result = {'contract_version': plan['contract_version'], 'input_preflight_passed': True,
              'training_cases': len(cases), 'new_samples_planned': 1152,
              'train_diagnostic_cases': len(suites['train_diagnostic']),
              'development_cases': len(suites['development']), 'max_prompt_tokens': max(map(len, prompts)),
              'max_logical_updates': 36, 'candidate_parent_step': summary['logical_optimizer_steps'],
              'seed_range': [seed_for(0, 0), seed_for(143, 7)],
              'parent_optimizer_sha256': plan['parent_optimizer_sha256'],
              'plan_sha256': file_hash(PLAN_PATH), 'source_files': len(source_identity['source_sha256']),
              'source_sha256': source_identity['source_sha256'],
              'gpu_runtime_checked': False, 'model_forward_calls': 0, 'optimizer_updates': 0,
              'cpu_weight_identity_checked': False}
    if check_weights:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from agentic_post_training.trainers.dpo import parameter_digest
        from agentic_post_training.trainers.grpo import lora_parameters
        torch.manual_seed(plan['seed'])
        model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation='eager')
        model = PeftModel.from_pretrained(model, str(parent / 'final-checkpoint/adapter'),
                                          is_trainable=True, autocast_adapter_dtype=False).eval()
        actual = parameter_digest(model)
        if actual != plan['parent_parameter_sha256']:
            raise ValueError('CPU parent parameter reconstruction differs')
        optimizer = torch.optim.AdamW(lora_parameters(model), lr=plan['learning_rate'], weight_decay=0.)
        bridge = import_parent_adam(model, optimizer, parent / 'final-checkpoint/optimizer.pt',
            expected_hash=plan['parent_optimizer_sha256'], old_binding=manifest['checkpoint_input_digest'],
            new_binding=digest_json({'preflight_only': file_hash(PLAN_PATH)}))
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError('preflight unexpectedly accumulated gradients')
        result.update(cpu_weight_identity_checked=True, reconstructed_parent_parameter_sha256=actual,
                      trainable_tensors=sum(p.requires_grad for p in model.parameters()),
                      trainable_parameters=sum(p.numel() for p in lora_parameters(model)),
                      adam_import=bridge)
    return result


def dispatch(price, invoke=subprocess.run):
    from agentic_post_training.rollouts.phase5_signal import run_seconds
    run_seconds(price)  # Validate cost input before creating any output.
    bundle = read(BUNDLE_MANIFEST)
    verify_files(ROOT, bundle['files'])
    verify_files(ROOT, bundle['required_existing_files'])
    plan = read(PLAN_PATH)
    source = ROOT / plan['source_rollout']
    old_archives = [ROOT / 'grpo-refresh-review-20261028.zip', ROOT / 'grpo-iteration-review-20261028.zip']
    if any(p.exists() for p in [source, OUTPUT, DRIVER, ARCHIVE, *old_archives]):
        raise FileExistsError('fresh iteration directories and archives required; no automatic retry')
    # Validate parent/base/data before starting any GPU subprocess.
    preflight_result = preflight()
    DRIVER.mkdir(parents=True, exist_ok=False)
    write(DRIVER / 'preflight.json', preflight_result)
    stages = []
    error = None
    try:
        for name, script in [('refresh', 'collect_grpo_iteration.py'), ('update', 'train_grpo_iteration.py')]:
            result = invoke([sys.executable, str(ROOT / 'scripts' / script), '--hourly-price', str(price)], cwd=ROOT)
            stages.append({'stage': name, 'exit_code': result.returncode})
            if result.returncode:
                raise RuntimeError(f'{name} failed with exit code {result.returncode}; subsequent stages not started')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        write(DRIVER / 'dispatch.json', {'stages': stages, 'error': error, 'automatic_retry': False,
              'instance_automatically_shutdown': False, 'max_seconds_per_phase': 1800,
              'max_estimated_yuan_per_phase': 10, 'hourly_price_yuan': price})
        with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED) as z:
            for folder in [source, OUTPUT, DRIVER]:
                if folder.exists():
                    for path in sorted(folder.rglob('*')):
                        if path.is_file():
                            z.write(path, path.relative_to(ROOT / 'artifacts').as_posix())
        print(f'Download from project root: {ARCHIVE}', flush=True)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hourly-price', type=float)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--check-cpu-weights', action='store_true')
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.preflight:
        result = preflight(check_weights=args.check_cpu_weights)
        if args.preflight_output:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            write(args.preflight_output, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        price = args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
        dispatch(price)


if __name__ == '__main__':
    main()
