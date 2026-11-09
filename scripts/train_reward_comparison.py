"""GPU preparation and independent matched reward arms, retaining serial GRPO gates."""

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import shutil

import torch
from reward_comparison_common import ROOT, OUTPUT, PLAN, ARMS, context, identity, load_shared, evaluation_metrics, import_comparison_adam
from collect_vllm_rollout import read, write
from explore_grpo_sampling import check_device, jsonl
from iteration_common import evaluation_suites
from graded_arithmetic_reward import grade_answer
from train_grpo_first import load_model, evaluate, predict, compare_evaluations
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.trainer_scores import score_response, no_update_metrics, sampler_trainer_difference
from agentic_post_training.trainers.dpo import parameter_digest
from agentic_post_training.trainers.grpo import (guarded_step, lora_parameters, save_optimizer_checkpoint,
    restore_optimizer_checkpoint, adapter_snapshot, compare_adapter_snapshot, optimizer_snapshot, compare_optimizer_snapshot)
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files, digest_json


def configure():
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def suites_for(ctx):
    return evaluation_suites(read(ROOT / 'configs/grpo_policy_iteration.json'))


def predictions(path, suites, tok):
    rows = jsonl(path)
    result = {name: [r for r in rows if r['suite'] == name] for name in suites}
    if len(rows) != sum(map(len, suites.values())):
        raise ValueError('complete evaluation predictions required')
    for name, cases in suites.items():
        if [r['case_id'] for r in result[name]] != [c.id for c in cases]:
            raise ValueError('prediction case order differs')
        for row, case in zip(result[name], cases):
            tokens = row['response_token_ids']
            if (not tokens or len(tokens) > 64 or tok.eos_token_id in tokens[:-1]
                    or (tokens[-1] == tok.eos_token_id) != (row['finish_reason'] == 'eos')
                    or row['text'] != tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)):
                raise ValueError('evaluation native tokens/finish/decode differs')
            reward = score_reward(row['text'], case, finish_reason=row['finish_reason'])
            if any(row[k] != v for k, v in reward.items()):
                raise ValueError('evaluation original reward differs')
    return result


def snapshot(folder, inv):
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / 'run_manifest.json', inv)
    for relative, digest in inv['source_sha256'].items():
        destination = folder / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if file_hash(destination) != digest:
            raise ValueError('source changed during snapshot')


def prepare(root):
    configure()
    ctx = context()
    plan, tok = ctx['plan'], ctx['tok']
    torch.manual_seed(plan['seed'])
    inv = {**identity(ctx), 'device': check_device(ctx['cfg'])}
    folder = root / 'shared'
    snapshot(folder, inv)
    policy = load_model(ctx['base'], ctx['parent'] / 'final-checkpoint/adapter', True)
    reference = load_model(ctx['base'], ctx['baseline'] / 'checkpoint/adapter', False)
    initial, ref_initial = parameter_digest(policy), parameter_digest(reference)
    if initial != plan['parent_parameter_sha256'] or ref_initial != plan['reference_parameter_sha256']:
        raise ValueError('parent/reference full parameter identity differs')
    suites = suites_for(ctx)
    sft = evaluate(reference, tok, suites, folder / 'sft_predictions.jsonl')
    parent = evaluate(policy, tok, suites, folder / 'parent_predictions.jsonl')
    for fresh, name in ((sft, 'sft_predictions.jsonl'), (parent, 'current_predictions.jsonl')):
        saved = predictions(ctx['parent'] / name, suites, tok)
        if any(a['response_token_ids'] != b['response_token_ids'] for suite in suites for a, b in zip(fresh[suite], saved[suite])):
            raise ValueError('fixed-start greedy outputs differ from audited parent/SFT')
    count = 0
    with (folder / 'trainer_scores.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for group in ctx['groups']:
            for sample in group:
                old = score_response(policy, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=False)
                new = score_response(policy, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=True)
                check = no_update_metrics(old, new, 1e-5)
                if not check['passed']:
                    raise ValueError('initial old/new probability gate failed')
                ref = score_response(reference, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=False)
                row = {'group_id': sample.group_id, 'sample_index': sample.sample_index,
                    'response_token_ids': sample.response_token_ids, 'behavior_token_log_probs': sample.behavior_token_log_probs,
                    'trainer_old_token_log_probs': old.cpu().tolist(), 'trainer_new_token_log_probs': new.detach().cpu().tolist(),
                    'reference_token_log_probs': ref.cpu().tolist(), 'no_update_check': check,
                    'sampler_trainer_difference': sampler_trainer_difference(sample.behavior_token_log_probs, old.cpu().tolist())}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                count += 1
                del old, new, ref
            stream.flush()
            print(f'Cached old/new/reference scores {count}/768', flush=True)
    unchanged = parameter_digest(policy) == initial and parameter_digest(reference) == ref_initial
    no_grad = all(p.grad is None for m in (policy, reference) for p in m.parameters())
    verify_files(ROOT, inv['source_sha256'])
    write(folder / 'checks.json', {'passed': unchanged and no_grad and count == 768,
        'initial_parameter_sha256': initial, 'reference_parameter_sha256': ref_initial,
        'parameters_unchanged': unchanged, 'no_grad_accumulated': no_grad, 'backward_calls': 0, 'optimizer_updates': 0,
        'records': count, 'start_greedy_matches_saved_parent_and_sft': True,
        'evaluation': {'sft': evaluation_metrics(sft, suites), 'parent': evaluation_metrics(parent, suites)},
        'files_sha256': {name: file_hash(folder / name) for name in ('trainer_scores.jsonl', 'sft_predictions.jsonl', 'parent_predictions.jsonl')}})
    if not unchanged or not no_grad:
        raise ValueError('shared preparation changed model or accumulated gradients')
    write(folder / 'summary.json', {'passed': True, 'input_identity_sha256': digest_json(inv),
        'files_sha256': {p.relative_to(folder).as_posix(): file_hash(p) for p in folder.rglob('*') if p.is_file()}})


def train_arm(root, arm):
    configure()
    ctx = context()
    plan, tok = ctx['plan'], ctx['tok']
    check_device(ctx['cfg'])
    shared, cached, reference_cache = load_shared(root, ctx)
    torch.manual_seed(plan['seed'])  # Identical independent arm RNG; never restore the parent's RNG.
    binding = digest_json({'shared': shared, 'arm': arm})
    inv = {**identity(ctx), 'device': 'NVIDIA L20', 'arm': arm, 'checkpoint_input_digest': binding,
           'shared_manifest_sha256': file_hash(root / 'shared/run_manifest.json')}
    output = root / arm
    snapshot(output, inv)
    policy = load_model(ctx['base'], ctx['parent'] / 'final-checkpoint/adapter', True)
    initial = parameter_digest(policy)
    frozen = parameter_digest(policy, frozen_only=True)
    if initial != plan['parent_parameter_sha256'] or sum(p.numel() for p in lora_parameters(policy)) != 230400:
        raise ValueError('arm did not start from the same 31-step policy')
    targets = {key: {**row, **ctx['targets'][arm][key]} for key, row in cached.items()}
    initial_error = 0.
    for group in ctx['groups']:
        for sample in group:
            fresh = score_response(policy, sample.prompt_token_ids, sample.response_token_ids, grad_enabled=False).cpu()
            old = torch.tensor(targets[sample.group_id, sample.sample_index]['trainer_old_token_log_probs'])
            initial_error = max(initial_error, (fresh - old).abs().max().item())
    if initial_error > plan['old_new_logp_tolerance']:
        raise ValueError('arm initial policy differs from common trainer old')
    optimizer = torch.optim.AdamW(lora_parameters(policy), lr=plan['learning_rate'], weight_decay=0.)
    lineage = import_comparison_adam(policy, optimizer, ctx['parent'] / 'final-checkpoint/optimizer.pt',
        expected_hash=plan['parent_optimizer_sha256'], old_binding=read(ctx['parent'] / 'run_manifest.json')['checkpoint_input_digest'],
        new_binding=binding)
    write(output / 'optimizer_lineage.json', lineage)
    indexed = {group[0].group_id: group for group in ctx['groups']}
    history, resume, stopped = [], None, None
    backward_calls = optimizer_calls = 0
    with (output / 'history.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for step, batch in enumerate(ctx['batches'], 1):
            samples = [sample for key in batch for sample in indexed[key]]
            metric = guarded_step(policy, optimizer, samples, targets, reference_cache, plan)
            backward_calls += len(samples)
            optimizer_calls += int(metric['updated'])
            if not metric['updated']:
                stopped = {**metric, 'step': step, 'group_ids': batch}
                stream.write(json.dumps(stopped, allow_nan=False) + '\n')
                break
            if step == 2:
                expected_adapter, expected_optimizer = adapter_snapshot(policy), optimizer_snapshot(optimizer)
                loss = metric['loss']
                del policy, optimizer
                gc.collect()
                torch.cuda.empty_cache()
                policy = load_model(ctx['base'], output / 'checkpoint-step1/adapter', True)
                optimizer = torch.optim.AdamW(lora_parameters(policy), lr=plan['learning_rate'], weight_decay=0.)
                local = restore_optimizer_checkpoint(policy, optimizer, output / 'checkpoint-step1', binding)
                if local != 1 or any(v['step'].item() != 32 for v in optimizer.state.values()):
                    raise ValueError('within-arm checkpoint local/cumulative step differs')
                resumed = guarded_step(policy, optimizer, samples, targets, reference_cache, plan)
                backward_calls += len(samples)
                optimizer_calls += int(resumed['updated'])
                resume = {**compare_adapter_snapshot(policy, expected_adapter, 1e-6),
                          **compare_optimizer_snapshot(optimizer, expected_optimizer, 1e-7),
                          'loss_abs_error': abs(resumed['loss'] - loss), 'restored_local_step': local, 'restored_adam_step': 32}
                resume['passed'] = bool(resume['passed'] and resume['optimizer_state_passed']
                                        and resume['loss_abs_error'] <= 1e-6 and resumed['updated'])
                write(output / 'resume_check.json', resume)
                if not resume['passed']:
                    raise ValueError('within-arm resume next step differs')
                metric = resumed
            metric.update(step=step, cumulative_candidate_step=31 + step, group_ids=batch)
            if any(v['step'].item() != 31 + step for v in optimizer.state.values()):
                raise ValueError('arm Adam counter differs')
            history.append(metric)
            stream.write(json.dumps(metric, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'arm': arm, **{k: metric[k] for k in ('step', 'loss', 'sampled_k3', 'ratio_clip_fraction')}}), flush=True)
            if step == 1:
                save_optimizer_checkpoint(policy, optimizer, output / 'checkpoint-step1', 1, binding)
    final = parameter_digest(policy)
    frozen_intact = parameter_digest(policy, frozen_only=True) == frozen and all(p.grad is None for p in policy.parameters() if not p.requires_grad)
    save_optimizer_checkpoint(policy, optimizer, output / 'final-checkpoint', len(history), binding)
    suites = suites_for(ctx)
    after = evaluate(policy, tok, suites, output / 'current_predictions.jsonl')
    samples = [s for group in ctx['groups'] for s in group][:8]
    scores = [score_response(policy, s.prompt_token_ids, s.response_token_ids, grad_enabled=False).cpu() for s in samples]
    del policy, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = load_model(ctx['base'], output / 'final-checkpoint/adapter', True)
    error = max((score_response(reloaded, s.prompt_token_ids, s.response_token_ids, grad_enabled=False).cpu() - old).abs().max().item()
                for s, old in zip(samples, scores))
    token_checks = [predict(reloaded, tok, suites['development'][i])['response_token_ids'] == after['development'][i]['response_token_ids']
                    for i in (0, 1, 2, 3, 52, 53, 54, 55)]
    reload_passed = error <= 1e-5 and all(token_checks) and parameter_digest(reloaded) == final
    write(output / 'reload_check.json', {'passed': reload_passed, 'max_abs_logp_error': error, 'greedy_token_checks': token_checks})
    verify_files(ROOT, inv['source_sha256'])
    # Revalidate old candidate, cache and all source identities after training, without a new forward pass.
    context()
    load_shared(root, ctx)
    before = predictions(root / 'shared/parent_predictions.jsonl', suites, tok)
    comparison = compare_evaluations(before, after)
    result = {'contract_version': plan['contract_version'], 'arm': arm, 'logical_optimizer_steps': len(history),
        'cumulative_candidate_steps': 31 + len(history), 'adam_step_at_start': 31, 'adam_step_at_end': 31 + len(history),
        'actual_optimizer_step_calls_including_discarded_probe': optimizer_calls, 'model_backward_calls': backward_calls,
        'early_stop': stopped, 'initial_parameter_sha256': initial, 'final_parameter_sha256': final,
        'frozen_parameters_unchanged': frozen_intact, 'policy_changed': final != initial,
        'initial_new_old_max_logp_error': initial_error, 'resume_next_step_passed': bool(resume and resume['passed']),
        'final_adapter_reload_passed': reload_passed, 'source_groups': 24, 'source_samples': 768,
        'evaluation': evaluation_metrics(after, suites), 'comparison_to_parent': comparison,
        'behavior_safety_gate_passed': all(v['new_structure_failure_cases'] == 0 for v in comparison.values())
            and comparison['development']['after_strict'] >= comparison['development']['before_strict'],
        'engineering_checks_passed': bool(frozen_intact and final != initial and reload_passed and resume and resume['passed']),
        'baseline_promoted': False, 'new_sampling_calls': 0, 'original_cross_engine_gate_passed': False,
        'monitor_note': 'token TIS and sampled k3 remain approximate; no post-update rollout refresh or automatic next training',
        'files_sha256': {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write(output / 'summary.json', result)
    if len(history) != 6 or not result['engineering_checks_passed']:
        raise RuntimeError('arm incomplete or engineering gate failed; stop the comparison')


def audit(root):
    ctx = context()
    shared, cached, refs = load_shared(root, ctx)
    suites = suites_for(ctx)
    parent = predictions(root / 'shared/parent_predictions.jsonl', suites, ctx['tok'])
    results = {}
    for arm in ARMS:
        folder = root / arm
        result, inv = read(folder / 'summary.json'), read(folder / 'run_manifest.json')
        verify_files(folder, result['files_sha256'])
        verify_files(folder / 'source', inv['source_sha256'])
        binding = digest_json({'shared': shared, 'arm': arm})
        history = jsonl(folder / 'history.jsonl')
        if (any(inv.get(k) != v for k, v in identity(ctx).items()) or inv['checkpoint_input_digest'] != binding
                or result['initial_parameter_sha256'] != ctx['plan']['parent_parameter_sha256']
                or not result['engineering_checks_passed'] or result['logical_optimizer_steps'] != 6
                or result['adam_step_at_end'] != 37 or result['model_backward_calls'] != 896
                or result['actual_optimizer_step_calls_including_discarded_probe'] != 7
                or len(history) != 6 or any(not h['updated'] or h['step'] != i or h['group_ids'] != ctx['batches'][i-1]
                                          or h['cumulative_candidate_step'] != 31+i for i, h in enumerate(history, 1))):
            raise ValueError('matched arm binding, step budget or history differs')
        checkpoint = torch.load(folder / 'final-checkpoint/optimizer.pt', map_location='cpu', weights_only=True)
        if (checkpoint['step'] != 6 or checkpoint['input_digest'] != binding
                or not checkpoint['optimizer']['state']
                or any(v['step'].item() != 37 for v in checkpoint['optimizer']['state'].values())):
            raise ValueError('arm final checkpoint counters/binding differ')
        after = predictions(folder / 'current_predictions.jsonl', suites, ctx['tok'])
        if evaluation_metrics(after, suites) != result['evaluation'] or compare_evaluations(parent, after) != result['comparison_to_parent']:
            raise ValueError('arm evaluation differs from recomputation')
        results[arm] = result['evaluation']
    comparison = {'contract_version': ctx['plan']['contract_version'], 'matched_comparison_completed': True,
        'logical_steps_per_arm': 6, 'same_parent_parameter_sha256': ctx['plan']['parent_parameter_sha256'],
        'same_samples_and_batches': True, 'parent': evaluation_metrics(parent, suites), 'arms': results,
        'graded_minus_equivalent_binary': {suite: {metric: results['graded'][suite][metric] - results['equivalent_binary'][suite][metric]
            for metric in ('strict', 'verified_correct', 'completed_semantic', 'arithmetic_graded_mean', 'length_stops')} for suite in suites},
        'baseline_promoted': False, 'automatic_winner_selection': False, 'gpu_reexecuted_by_audit': False}
    write(root / 'comparison.json', comparison)
    return comparison


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('prepare', *ARMS, 'audit'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.output)
    elif args.stage == 'audit': print(json.dumps(audit(args.output), ensure_ascii=False), flush=True)
    else: train_arm(args.output, args.stage)
