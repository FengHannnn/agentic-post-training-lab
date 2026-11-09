"""One refreshed-policy iteration with fixed SFT reference and inherited Adam."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from collect_grpo_iteration import load_run, audit, run_seconds
from iteration_common import ROOT, PLAN_PATH, read, identity, check_device, evaluation_suites
from collect_vllm_rollout import write
from train_grpo_first import load_model, predict, evaluate, compare_evaluations
from agentic_post_training.rollouts.trainer_scores import score_response
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (
    guarded_step, lora_parameters, save_optimizer_checkpoint, restore_optimizer_checkpoint,
    adapter_snapshot, compare_adapter_snapshot, optimizer_snapshot, compare_optimizer_snapshot)
from agentic_post_training.trainers.policy_iteration import (
    digest_json, file_hash, verify_files, verify_parent, select_mixed_batches, import_parent_adam)

ARCHIVE = ROOT / 'grpo-iteration-review-20261028.zip'


def context():
    plan = read(PLAN_PATH)
    source = ROOT / plan['source_rollout']
    summary = audit(source, write_summary=False)
    if not summary['signal_collection_interface_passed']:
        raise ValueError('audited fresh candidate rollout required')
    verify_files(source, summary['files_sha256'])
    cfg, plan, baseline, base, tok, rows, groups, packed, rewards = load_run(source)
    if check_device(cfg) != read(source / 'run_manifest.json')['device']:
        raise ValueError('fresh scoring GPU differs')
    records = [json.loads(line) for line in (source / 'trainer_scores.jsonl').read_text(encoding='utf-8').splitlines()]
    targets = {(r['group_id'], r['sample_index']): r for r in records}
    if len(targets) != 1152:
        raise ValueError('all unique fresh trainer records required')
    batches = select_mixed_batches(rewards, plan['case_ids'])
    if len(batches) > plan['max_logical_steps']:
        raise ValueError('precommitted maximum steps exceeded')
    return plan, source, cfg, baseline, base, tok, groups, targets, evaluation_suites(plan), batches


def compare_three(sft, previous, current):
    comparisons = {'sft_to_previous': compare_evaluations(sft, previous),
                   'previous_to_current': compare_evaluations(previous, current),
                   'sft_to_current': compare_evaluations(sft, current)}
    for name, suites in comparisons.items():
        for suite, result in suites.items():
            result['strict_improved_case_ids'] = [r['case_id'] for r in result['changed_cases']
                                                  if not r['before_strict'] and r['after_strict']]
            result['strict_regressed_case_ids'] = [r['case_id'] for r in result['changed_cases']
                                                   if r['before_strict'] and not r['after_strict']]
    return comparisons


def train(output):
    import torch

    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan, source, cfg, baseline, base, tok, groups, targets, suites, batches = context()
    torch.manual_seed(plan['seed'])
    parent, parent_summary, parent_manifest = verify_parent(ROOT, plan)
    inv = identity(cfg, plan, baseline)
    binding = digest_json({'identity': inv, 'source_summary_sha256': file_hash(source / 'summary.json'),
                           'source_manifest_sha256': file_hash(source / 'run_manifest.json'),
                           'group_batches': batches})
    output.mkdir(parents=True, exist_ok=False)
    inv.update(device=check_device(cfg), source_summary_sha256=file_hash(source / 'summary.json'),
               source_manifest_sha256=file_hash(source / 'run_manifest.json'),
               checkpoint_input_digest=binding, group_batches=batches)
    write(output / 'run_manifest.json', inv)
    for relative, digest in inv['source_sha256'].items():
        dest = output / 'source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
        if file_hash(dest) != digest:
            raise ValueError('source changed while taking snapshot')

    reference = load_model(base, baseline / 'checkpoint/adapter', False)
    ref_initial = parameter_digest(reference)
    if ref_initial != inv['policy_version']['reference']:
        raise ValueError('original frozen SFT reference identity differs')
    sft = evaluate(reference, tok, suites, output / 'sft_predictions.jsonl')
    policy = load_model(base, parent / 'final-checkpoint/adapter', True)
    initial = parameter_digest(policy)
    if initial != plan['parent_parameter_sha256'] or sum(p.numel() for p in lora_parameters(policy)) != 230400:
        raise ValueError('new policy did not start from parent candidate')
    frozen_initial = parameter_digest(policy, frozen_only=True)
    indexed = {g[0].group_id: g for g in groups}
    selected = [s for batch in batches for key in batch for s in indexed[key]]
    # Validate every stored old score, even for zero-variance groups.
    reference_cache = {}
    initial_error = 0.
    for g, group in enumerate(groups):
        for sample in group:
            key = sample.group_id, sample.sample_index
            fresh = score_response(policy, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=False).cpu()
            old = torch.tensor(targets[key]['trainer_old_token_log_probs'], dtype=torch.float32)
            initial_error = max(initial_error, (fresh - old).abs().max().item())
            reference_cache[key] = score_response(reference, sample.prompt_token_ids,
                                                   sample.response_token_ids, grad_enabled=False).cpu()
        if (g + 1) % 12 == 0:
            print(f'Checked new old/reference groups {g + 1}/144', flush=True)
    if initial_error > plan['old_new_logp_tolerance']:
        raise ValueError('initial new differs from this iteration old; reference is checked independently')
    torch.save({f'{s.group_id}/{s.sample_index}': reference_cache[s.group_id, s.sample_index]
                for group in groups for s in group}, output / 'reference_scores.pt')
    before = evaluate(policy, tok, suites, output / 'previous_predictions.jsonl')
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=plan['learning_rate'], weight_decay=0.)
    bridge = import_parent_adam(policy, optimizer, parent / 'final-checkpoint/optimizer.pt',
        expected_hash=plan['parent_optimizer_sha256'], old_binding=parent_manifest['checkpoint_input_digest'],
        new_binding=binding)
    write(output / 'optimizer_lineage.json', bridge)
    history = []
    optimizer_calls = backward_calls = 0
    stopped = None
    resume_check = None
    with (output / 'history.jsonl').open('x', encoding='utf-8', newline='\n') as log:
        for step, batch_ids in enumerate(batches, 1):
            samples = [s for key in batch_ids for s in indexed[key]]
            metric = guarded_step(policy, optimizer, samples, targets, reference_cache, plan)
            backward_calls += len(samples)
            metric.update(step=step, cumulative_candidate_step=16 + step, group_ids=batch_ids)
            if not metric['updated']:
                stopped = metric
                log.write(json.dumps(metric, allow_nan=False) + '\n')
                log.flush()
                break
            optimizer_calls += 1
            if step == 2:
                expected_adapter = adapter_snapshot(policy)
                expected_optimizer = optimizer_snapshot(optimizer)
                uninterrupted = metric.copy()
                del policy, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                policy = load_model(base, output / 'checkpoint-step1/adapter', True)
                optimizer = torch.optim.AdamW(lora_parameters(policy), lr=plan['learning_rate'], weight_decay=0.)
                restored_step = restore_optimizer_checkpoint(policy, optimizer, output / 'checkpoint-step1', binding)
                if restored_step != 1:
                    raise ValueError('within-iteration logical checkpoint step differs')
                resumed = guarded_step(policy, optimizer, samples, targets, reference_cache, plan)
                backward_calls += len(samples)
                optimizer_calls += int(resumed['updated'])
                resume_check = {**compare_adapter_snapshot(policy, expected_adapter, 1e-6),
                                **compare_optimizer_snapshot(optimizer, expected_optimizer, 1e-7),
                                'uninterrupted_loss': uninterrupted['loss'], 'restored_loss': resumed['loss'],
                                'restored_local_step': restored_step, 'restored_adam_step': 17}
                resume_check['passed'] = bool(resume_check['passed'] and resume_check['optimizer_state_passed']
                    and resumed['updated'] and abs(uninterrupted['loss'] - resumed['loss']) <= 1e-6)
                write(output / 'resume_check.json', resume_check)
                if not resume_check['passed']:
                    raise ValueError('within-iteration resume next step differs')
                metric = {**resumed, 'step': step, 'cumulative_candidate_step': 16 + step, 'group_ids': batch_ids}
            history.append(metric)
            if any(v['step'].item() != 16 + len(history) for v in optimizer.state.values()):
                raise ValueError('inherited Adam cumulative step differs')
            log.write(json.dumps(metric, allow_nan=False) + '\n')
            log.flush()
            print(json.dumps({k: metric[k] for k in ('step', 'cumulative_candidate_step', 'loss', 'sampled_k3',
                                                    'ratio_clip_fraction', 'grad_norm_before_clip')}), flush=True)
            if step == 1:
                save_optimizer_checkpoint(policy, optimizer, output / 'checkpoint-step1', 1, binding)

    final = parameter_digest(policy)
    frozen_intact = parameter_digest(policy, frozen_only=True) == frozen_initial
    reference_intact = parameter_digest(reference) == ref_initial and all(
        not p.requires_grad and p.grad is None for p in reference.parameters())
    if not frozen_intact or not reference_intact or (history and final == initial):
        raise ValueError('frozen/reference changed or attempted updates did not change policy')
    if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
        raise ValueError('frozen policy received gradients')
    save_optimizer_checkpoint(policy, optimizer, output / 'final-checkpoint', len(history), binding)
    tok.save_pretrained(output / 'tokenizer')
    after = evaluate(policy, tok, suites, output / 'current_predictions.jsonl')
    probes = (selected or [s for group in groups for s in group])[:8]
    scores = [score_response(policy, s.prompt_token_ids, s.response_token_ids, grad_enabled=False).cpu() for s in probes]
    del policy, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_model(base, output / 'final-checkpoint/adapter', True)
    max_error = max((score_response(reloaded, s.prompt_token_ids, s.response_token_ids,
                                  grad_enabled=False).cpu() - expected).abs().max().item()
                    for s, expected in zip(probes, scores))
    token_checks = [predict(reloaded, tok, suites['development'][i])['response_token_ids']
                    == after['development'][i]['response_token_ids'] for i in (0, 1, 2, 3, 52, 53, 54, 55)]
    reload_passed = max_error <= 1e-5 and all(token_checks) and parameter_digest(reloaded) == final
    write(output / 'reload_check.json', {'passed': reload_passed, 'max_abs_logp_error': max_error,
                                         'greedy_token_checks': token_checks})
    # Source and input artifacts must still match after the GPU work.
    verify_files(ROOT, inv['source_sha256'])
    verify_files(source, read(source / 'summary.json')['files_sha256'])
    if file_hash(source / 'summary.json') != inv['source_summary_sha256']:
        raise ValueError('fresh rollout summary changed during training')
    verify_parent(ROOT, plan)
    comparison = compare_three(sft, before, after)
    safety = all(all(r['new_structure_failure_cases'] == 0 for r in comparison[key].values())
                 and comparison[key]['development']['after_strict'] >= comparison[key]['development']['before_strict']
                 for key in ('previous_to_current', 'sft_to_current'))
    engineering = bool(frozen_intact and reference_intact and reload_passed
                       and (len(history) < 2 or (resume_check and resume_check['passed'])))
    result = {'contract_version': plan['contract_version'], 'iteration': 2,
              'policy_version': inv['policy_version'], 'optimizer_mode': plan['optimizer_mode'],
              'logical_optimizer_steps': len(history), 'cumulative_candidate_steps': 16 + len(history),
              'adam_step_at_start': 16, 'adam_step_at_end': 16 + len(history),
              'actual_optimizer_step_calls_including_discarded_probe': optimizer_calls,
              'model_backward_calls': backward_calls, 'available_mixed_groups': sum(map(len, batches)),
              'processed_candidate_groups': sum(len(r['group_ids']) for r in history),
              'source_groups': 144, 'source_samples': 1152,
              'status': 'updated' if history else 'no_signal' if not batches else 'stopped_before_update',
              'early_stop': stopped, 'initial_parameter_sha256': initial, 'final_parameter_sha256': final,
              'initial_new_old_max_logp_error': initial_error,
              'reference_parameter_sha256': ref_initial, 'reference_equals_old': ref_initial == initial,
              'frozen_parameters_unchanged': frozen_intact, 'reference_unchanged_and_no_grad': reference_intact,
              'resume_next_step_passed': resume_check['passed'] if resume_check else None,
              'resume_not_applicable_reason': None if resume_check else 'fewer_than_two_completed_updates',
              'final_adapter_reload_passed': reload_passed, 'policy_changed': final != initial,
              'engineering_checks_passed': engineering, 'behavior_safety_gate_passed': safety,
              'development_strict_net_gain_vs_previous': comparison['previous_to_current']['development']['after_strict']
                  - comparison['previous_to_current']['development']['before_strict'],
              'evaluation': comparison, 'baseline_promoted': False, 'rollout_refreshed_after_update': True,
              'original_cross_engine_gate_passed': False,
              'monitor_note': 'token TIS is approximate; sampled k3 is not an unbiased full-policy KL',
              'files_sha256': {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write(output / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files_sha256', 'evaluation')}), flush=True)
    if not engineering:
        raise RuntimeError('iteration engineering checks failed')


def run(args):
    if args.output.exists() or ARCHIVE.exists():
        raise FileExistsError('fresh output/archive required; no automatic retry')
    price = args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
    seconds = run_seconds(price)
    started = time.perf_counter()
    error = None
    try:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--stage', 'train',
                                 '--output', str(args.output.resolve())], cwd=ROOT, start_new_session=(os.name != 'nt'))
        try:
            code = proc.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            if os.name != 'nt':
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != 'nt':
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait()
            raise TimeoutError('bounded iteration stopped; no automatic retry')
        if code:
            raise RuntimeError(f'iteration exited {code}')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        elapsed = time.perf_counter() - started
        write(args.output / 'cost_and_run.json', {'elapsed_run_seconds': elapsed, 'hourly_price_yuan': price,
              'estimated_run_cost_yuan': elapsed * price / 3600, 'error': error, 'max_run_seconds': seconds,
              'billing_scope': 'job only; excludes boot/idle/storage/rounding', 'instance_automatically_shutdown': False})
        with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(args.output.rglob('*')):
                if p.is_file():
                    z.write(p, str(p.relative_to(args.output.parent)))
        print(f'Download from project root: {ARCHIVE}', flush=True)
    if error:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('run', 'train'), default='run')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/checkpoints/grpo-iteration-20261028-01')
    parser.add_argument('--hourly-price', type=float)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.stage == 'run':
        run(args)
    else:
        train(args.output)


if __name__ == '__main__':
    main()
