"""Bindings and finite reward admission for a matched two-arm reward experiment."""

from dataclasses import asdict
import math
from pathlib import Path

import torch
import explore_grpo_sampling as exploration
from collect_vllm_rollout import ROOT, read
from graded_arithmetic_reward import CONTRACT, LEVELS, grade_answer
from agentic_post_training.algorithms.losses import group_relative_advantages
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rollouts.trainer_scores import no_update_metrics
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files, digest_json
from agentic_post_training.trainers.grpo import compare_optimizer_snapshot

PLAN = ROOT / 'configs/experiments/grpo_reward_comparison.json'
OUTPUT = ROOT / 'artifacts/checkpoints/reward-comparison-20261028-01'
ARMS = ('equivalent_binary', 'graded')
NEW_FILES = ('scripts/reward_comparison_common.py', 'scripts/train_reward_comparison.py',
             'scripts/run_reward_comparison.py', 'scripts/graded_arithmetic_reward.py',
             'configs/experiments/grpo_reward_comparison.json')


def admit_rewards(values, arm):
    allowed = (0., 1.) if arm == 'equivalent_binary' else tuple(LEVELS.values()) if arm == 'graded' else ()
    if len(values) != 32 or any(type(v) not in (int, float) or not math.isfinite(v) or v not in allowed for v in values):
        raise ValueError('32 finite rewards under the declared arm contract required')
    return group_relative_advantages(torch.tensor([values], dtype=torch.float64))[0].tolist()


def import_comparison_adam(policy, optimizer, checkpoint, *, expected_hash, old_binding, new_binding,
                           parent_local_step=15, parent_adam_step=31):
    if old_binding == new_binding or file_hash(Path(checkpoint)) != expected_hash:
        raise ValueError('distinct new binding and verified parent optimizer required')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    state = saved['optimizer']
    if (saved['input_digest'] != old_binding or saved['step'] != parent_local_step
            or saved['trainable_names'] != [n for n, _ in named] or not named
            or any('lora_' not in n for n, _ in named)
            or len(state['param_groups']) != 1 or len(optimizer.param_groups) != 1):
        raise ValueError('parent binding, local step or parameter order differs')
    group, current = state['param_groups'][0], optimizer.state_dict()['param_groups'][0]
    if (len(group['params']) != len(named) or len(set(group['params'])) != len(named)
            or set(state['state']) != set(group['params'])):
        raise ValueError('parent Adam coverage differs')
    for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
        if group[key] != current[key]:
            raise ValueError('parent Adam hyperparameter differs: ' + key)
    for index, (_, parameter) in zip(group['params'], named):
        values = state['state'][index]
        if set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('unexpected Adam buffers')
        step = values['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != parent_adam_step:
            raise ValueError('parent cumulative Adam step differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = values[key]
            if (value.shape != parameter.shape or value.dtype != parameter.dtype or not torch.isfinite(value).all()
                    or (key == 'exp_avg_sq' and (value < 0).any())):
                raise ValueError('invalid parent Adam moments')
    optimizer.load_state_dict(state)
    comparison = compare_optimizer_snapshot(optimizer, state['state'], 0.)
    if not comparison['optimizer_state_passed']:
        raise ValueError('parent Adam import changed moments')
    return {'mode': 'carry_parent_adam', 'parent_local_step': parent_local_step,
            'adam_step_at_start': parent_adam_step, 'parent_rng_restored': False,
            'parent_optimizer_sha256': expected_hash, 'old_binding': old_binding, 'new_binding': new_binding, **comparison}


def context():
    plan = read(PLAN)
    fixed = {'contract_version': 'grpo-reward-comparison-v1', 'arms': list(ARMS), 'max_logical_steps_per_arm': 6,
             'groups_per_step': 4, 'samples_per_group': 32, 'learning_rate': 1e-5, 'beta': .02,
             'clip_epsilon': .2, 'tis_cap': 2., 'max_grad_norm': 1., 'stop_sampled_k3': .05,
             'stop_ratio_clip_fraction': .5, 'old_new_logp_tolerance': 1e-5, 'reload_logp_tolerance': 1e-5,
             'seed': 202610290, 'parent_local_step': 15, 'parent_adam_step': 31,
             'max_wall_seconds': 1800, 'max_run_cost_yuan': 10., 'auto_promote_baseline': False,
             'reward_scorer_contract': CONTRACT, 'reward_levels': LEVELS}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('fixed comparison contract differs')
    original = exploration.audit(exploration.OUTPUT, write_summary=False)
    ctx = exploration.setup()
    source_plan, cfg, parent, baseline, base, tok, selected, cases, prompts = ctx
    if (file_hash(exploration.OUTPUT / 'summary.json') != plan['source_summary_sha256']
            or original['candidate_parameter_sha256'] != plan['parent_parameter_sha256']
            or source_plan['candidate_summary_sha256'] != plan['parent_summary_sha256']
            or file_hash(parent / 'final-checkpoint/optimizer.pt') != plan['parent_optimizer_sha256']
            or file_hash(ROOT / 'scripts/graded_arithmetic_reward.py') != plan['reward_scorer_sha256']
            or read(ROOT / 'configs/trainer_probability_contract.json')['frozen_fp32_parameter_sha256'] != plan['reference_parameter_sha256']
            or plan['case_ids'] != source_plan['case_ids']):
        raise ValueError('parent, scorer or temperature-one source identity differs')
    rows = [r for r in exploration.jsonl(exploration.OUTPUT / 'rollouts.jsonl') if r['arm'] == 't100']
    indexed = {(r['case_id'], r['sample_index']): r for r in rows}
    if len(rows) != len(indexed) or len(rows) != 768:
        raise ValueError('all 768 temperature-one records required')
    groups, grades = [], {}
    for case in cases:
        group = []
        for index in range(32):
            row = indexed[case.id, index]
            if row['temperature'] != 1. or row['behavior_token_log_probs'] != row['sampler_raw_model_token_log_probs']:
                raise ValueError('temperature-one behavior semantics required')
            group.append(RolloutSample(case.id, case.id, index, row['seed'], row['prompt_token_ids'],
                row['response_token_ids'], row['sampler_raw_model_token_log_probs'], row['behavior_token_log_probs'],
                row['eos_token_id'], row['finish_reason'], row['text'], row['raw_text']))
            grades[case.id, index] = grade_answer(row['text'], case, finish_reason=row['finish_reason'])
        groups.append(group)
    pack_rollout_groups(groups, tok.pad_token_id)
    batches = [plan['case_ids'][i:i+4] for i in range(0, 24, 4)]
    if len(batches) != 6 or any(len(b) != 4 for b in batches):
        raise ValueError('six matched four-group batches required')
    targets = {}
    for arm in ARMS:
        field = 'verified_equivalent_reward' if arm == 'equivalent_binary' else 'graded_reward'
        targets[arm] = {}
        for group in groups:
            values = [grades[s.case_id, s.sample_index][field] for s in group]
            advantages = admit_rewards(values, arm)
            for sample, reward, advantage in zip(group, values, advantages):
                targets[arm][sample.case_id, sample.sample_index] = {'reward': reward, 'advantage': advantage}
    return {'plan': plan, 'cfg': cfg, 'parent': parent, 'baseline': baseline, 'base': base, 'tok': tok,
            'groups': groups, 'grades': grades, 'targets': targets, 'batches': batches, 'cases': cases}


def identity(ctx):
    sources = read(exploration.OUTPUT / 'run_manifest.json')['source_sha256'].copy()
    sources.update({name: file_hash(ROOT / name) for name in NEW_FILES})
    return {'plan': ctx['plan'], 'source_sha256': sources,
            'parent_summary_sha256': file_hash(ctx['parent'] / 'summary.json'),
            'source_summary_sha256': file_hash(exploration.OUTPUT / 'summary.json'),
            'baseline_manifest_sha256': file_hash(ctx['baseline'] / 'manifest.json'),
            'data_manifest_sha256': file_hash(ROOT / 'data/preferences/dpo_v2/manifest.json'),
            'group_batches': ctx['batches'], 'new_sampling_calls': 0}


def evaluation_metrics(predictions, suites):
    metrics = {}
    for suite, rows in predictions.items():
        if [r['case_id'] for r in rows] != [c.id for c in suites[suite]]:
            raise ValueError('evaluation case order differs')
        graded = [grade_answer(r['text'], c, finish_reason=r['finish_reason']) for r, c in zip(rows, suites[suite])]
        metrics[suite] = {'cases': len(rows), 'strict': sum(r['strict_pass'] and r['finish_reason'] == 'eos' for r in rows),
            'verified_correct': sum(g['verified_equivalent_reward'] if g['graded_eligible'] else
                                    float(r['strict_pass'] and r['finish_reason'] == 'eos') for r, g in zip(rows, graded)),
            'completed_semantic': sum(r['semantic_pass'] and r['finish_reason'] == 'eos' for r in rows),
            'arithmetic_graded_mean': sum(g['graded_reward'] for g in graded if g['graded_eligible']) / sum(g['graded_eligible'] for g in graded),
            'length_stops': sum(r['finish_reason'] == 'length' for r in rows)}
    return metrics


def load_shared(root, ctx):
    folder = Path(root) / 'shared'
    inv = read(folder / 'run_manifest.json')
    summary = read(folder / 'summary.json')
    verify_files(folder, summary['files_sha256'])
    if not summary['passed'] or summary['input_identity_sha256'] != digest_json(inv):
        raise ValueError('prepared source/cache summary differs')
    if any(inv.get(k) != v for k, v in identity(ctx).items()):
        raise ValueError('shared input binding differs')
    verify_files(folder / 'source', inv['source_sha256'])
    checks = read(folder / 'checks.json')
    if (not checks['passed'] or checks['initial_parameter_sha256'] != ctx['plan']['parent_parameter_sha256']
            or checks['reference_parameter_sha256'] != ctx['plan']['reference_parameter_sha256']
            or not checks['parameters_unchanged'] or not checks['no_grad_accumulated']
            or checks['optimizer_updates'] != 0 or checks['backward_calls'] != 0):
        raise ValueError('shared no-update checks differ')
    verify_files(folder, checks['files_sha256'])
    records = exploration.jsonl(folder / 'trainer_scores.jsonl')
    flat = [s for group in ctx['groups'] for s in group]
    if len(records) != len(flat):
        raise ValueError('768 cached old/reference scores required')
    targets, reference = {}, {}
    for sample, row in zip(flat, records):
        key = sample.group_id, sample.sample_index
        if ((row['group_id'], row['sample_index']) != key or row['response_token_ids'] != sample.response_token_ids
                or row['behavior_token_log_probs'] != sample.behavior_token_log_probs):
            raise ValueError('cached probability sample identity differs')
        old = torch.tensor(row['trainer_old_token_log_probs'], dtype=torch.float32)
        new = torch.tensor(row['trainer_new_token_log_probs'], dtype=torch.float32, requires_grad=True)
        ref = torch.tensor(row['reference_token_log_probs'], dtype=torch.float32)
        if (old.numel() != len(sample.response_token_ids) or ref.shape != old.shape or not torch.isfinite(ref).all()
                or (ref > 1e-6).any() or no_update_metrics(old, new, 1e-5) != row['no_update_check']
                or not row['no_update_check']['passed']):
            raise ValueError('cached old/new/reference probabilities differ')
        targets[key], reference[key] = row, ref
    return {**inv, 'preparation_summary_sha256': file_hash(folder / 'summary.json')}, targets, reference
