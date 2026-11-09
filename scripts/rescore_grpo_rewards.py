"""Verify and rescore existing answers without generation, training or historical mutation."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from agentic_post_training.algorithms.losses import group_relative_advantages
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rewards.audit import audit_records
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files
from graded_arithmetic_reward import CONTRACT, LEVELS, grade_answer
from diagnose_grpo_effectiveness import read, jsonl, write, write_rows

PLAN = ROOT / 'configs/experiments/grpo_graded_reward_rescore.json'
OUTPUT = ROOT / 'artifacts/evaluations/grpo-graded-reward-rescore-20261028-01'
VARIANTS = {'strict': 'strict_reward', 'equivalent_binary': 'verified_equivalent_reward', 'graded': 'graded_reward'}


def group_statistics(rows):
    if len(rows) < 2 or [r['sample_index'] for r in rows] != list(range(len(rows))):
        raise ValueError('complete ordered group with at least two samples required')
    result = {}
    for variant, key in VARIANTS.items():
        rewards = torch.tensor([[r[key] for r in rows]], dtype=torch.float64)
        advantages = group_relative_advantages(rewards)[0].tolist()
        result[variant] = {'rewards': rewards[0].tolist(), 'advantages': advantages,
            'mean': rewards.mean().item(), 'std': rewards.std(unbiased=False).item(),
            'has_relative_signal': rewards.std(unbiased=False).item() > 1e-8,
            'nonzero_advantage_samples': sum(abs(a) > 1e-8 for a in advantages)}
    return result


def aggregate(groups):
    if not groups:
        raise ValueError('nonempty group slice required')
    result = {'groups': len(groups), 'samples': sum(g['samples'] for g in groups), 'variants': {}}
    for variant in VARIANTS:
        result['variants'][variant] = {'nonzero_variance_groups': sum(g['statistics'][variant]['has_relative_signal'] for g in groups),
            'nonzero_advantage_samples': sum(g['statistics'][variant]['nonzero_advantage_samples'] for g in groups),
            'reward_sum': sum(sum(g['statistics'][variant]['rewards']) for g in groups)}
    result.update({
        'groups_with_verified_full_credit': sum(any(r == 1. for r in g['statistics']['equivalent_binary']['rewards']) for g in groups),
        'new_signal_strict_to_equivalent': sum(not g['statistics']['strict']['has_relative_signal'] and g['statistics']['equivalent_binary']['has_relative_signal'] for g in groups),
        'lost_signal_strict_to_equivalent': sum(g['statistics']['strict']['has_relative_signal'] and not g['statistics']['equivalent_binary']['has_relative_signal'] for g in groups),
        'new_signal_equivalent_to_graded': sum(not g['statistics']['equivalent_binary']['has_relative_signal'] and g['statistics']['graded']['has_relative_signal'] for g in groups),
        'partial_only_signal_groups': sum(g['statistics']['graded']['has_relative_signal'] and max(g['statistics']['graded']['rewards']) < 1. for g in groups),
        'still_zero_variance_groups': sum(not g['statistics']['graded']['has_relative_signal'] for g in groups)})
    return result


def summarize_rollouts(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['kind'] == 'rollout':
            grouped[row['source'], row['case_id']].append(row)
    groups = []
    for (source, case_id), values in sorted(grouped.items()):
        values.sort(key=lambda r: r['sample_index'])
        size = 32 if source.startswith('exploration_') else 8
        if len(values) != size:
            raise ValueError('planned group size differs')
        for prefix in ([8, 16, 32] if size == 32 else [8]):
            part = values[:prefix]
            groups.append({'source': source, 'prefix': prefix, 'case_id': case_id,
                **{k: part[0][k] for k in ('wording', 'category', 'family_id', 'role')},
                'samples': prefix, 'statistics': group_statistics(part)})
    summaries = {}
    for source, prefix in sorted({(g['source'], g['prefix']) for g in groups}):
        part = [g for g in groups if (g['source'], g['prefix']) == (source, prefix)]
        summaries[f'{source}/prefix{prefix}'] = {'overall': aggregate(part),
            'by_wording': {v: aggregate([g for g in part if g['wording'] == v]) for v in sorted({g['wording'] for g in part})},
            'by_role': {v: aggregate([g for g in part if g['role'] == v]) for v in sorted({g['role'] for g in part})}}
    return groups, summaries


def verified_inputs():
    import explore_grpo_sampling as exploration
    from iteration_common import setup
    plan = read(PLAN)
    if (plan['contract_version'] != CONTRACT or plan['levels'] != LEVELS or plan['new_model_forwards'] != 0
            or plan['optimizer_updates'] != 0 or plan['training_admitted']):
        raise ValueError('graded offline contract differs')
    # These use the original pinned source inventories; new scripts/nested config do not enter them.
    _, iteration, baseline, base, tok, cases, prompts = setup()
    files = {}
    for item in plan['input_runs']:
        folder = ROOT / item['path']
        if file_hash(folder / 'summary.json') != item['summary_sha256']:
            raise ValueError('input summary identity differs')
        summary = read(folder / 'summary.json')
        verify_files(folder, summary['files_sha256'])
        inv = read(folder / 'run_manifest.json')
        verify_files(folder / 'source', inv['source_sha256'])
        verify_files(ROOT, inv['source_sha256'])
        for name in ('summary.json', 'run_manifest.json', *summary['files_sha256']):
            path = folder / name
            files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    exploration.audit(exploration.OUTPUT, write_summary=False)
    folder = ROOT / 'data/preferences/dpo_v2'
    manifest = read(folder / 'manifest.json')
    current = ROOT / 'artifacts/checkpoints/grpo-iteration-20261028-01'
    if file_hash(folder / 'manifest.json') != read(current / 'run_manifest.json')['data_manifest_sha256']:
        raise ValueError('preference manifest identity differs')
    splits = load_preference_splits(folder, manifest)
    examples = {r.id: r for values in splits.values() for r in values}
    for name in ('manifest.json', *manifest['files']):
        path = folder / name
        files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for path in (PLAN, Path(__file__), ROOT / 'scripts/graded_arithmetic_reward.py'):
        files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return plan, iteration, tok, cases, prompts, examples, files


def gather_rows(context):
    plan, iteration, tok, cases, prompts, examples, files = context
    result = []
    def append(sample, example, *, source, kind, suite, index, role, original):
        case = example.target()
        if sample['text'] != tok.decode(sample['response_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False):
            raise ValueError('saved native decode differs')
        old = score_reward(sample['text'], case, finish_reason=sample['finish_reason'])
        if kind == 'prediction':
            if any(original[k] != value for k, value in old.items()):
                raise ValueError('saved prediction strict scoring differs')
        elif original['score'] != old['score'] or original['semantic_pass'] != old['semantic_pass']:
            raise ValueError('saved rollout scoring differs')
        graded = grade_answer(sample['text'], case, finish_reason=sample['finish_reason'])
        if graded['strict_reward'] != old['reward'] or graded['verified_equivalent_reward'] < old['reward']:
            raise ValueError('new grading lost strict-positive answer')
        result.append({'source': source, 'kind': kind, 'suite': suite, 'case_id': case.id,
            'family_id': example.family_id, 'wording': example.provenance['wording'], 'category': example.category,
            'role': role, 'sample_index': index, 'text': sample['text'], 'finish_reason': sample['finish_reason'],
            'response_token_ids': sample['response_token_ids'], 'semantic_completed': bool(old['semantic_pass'] and sample['finish_reason'] == 'eos'),
            **graded})
    for label, path, seed in [('sft_rollout', 'artifacts/trajectories/phase5-signal-20261021-01', 20261021),
                              ('previous_rollout', 'artifacts/trajectories/grpo-refresh-20261028-01', 202610280)]:
        folder = ROOT / path
        rows = jsonl(folder / 'rollouts.jsonl')
        if len(rows) != 1152 or audit_records(rows) != read(folder / 'reward_audit.json'):
            raise ValueError('complete historical reward audit required')
        for position, row in enumerate(rows):
            group, index = divmod(position, 8)
            sample, case = row['sample'], cases[group]
            if (row['case'] != asdict(case) or sample['case_id'] != case.id or sample['seed'] != seed + group * 1000 + index
                    or sample['prompt_token_ids'] != prompts[group]):
                raise ValueError('historical sample identity differs')
            append(sample, examples[case.id], source=label, kind='rollout', suite='train', index=index, role='full_train', original=row)
    exploration = ROOT / 'artifacts/trajectories/grpo-exploration-20261028-01'
    original = read(exploration / 'summary.json')
    roles = {r['case_id']: r['role'] for r in original['by_arm_and_prefix']['t100']['32']['per_case']}
    for row in jsonl(exploration / 'rollouts.jsonl'):
        append(row, examples[row['case_id']], source='exploration_' + row['arm'], kind='rollout', suite='train',
               index=row['sample_index'], role=roles[row['case_id']], original=row)
    current = ROOT / 'artifacts/checkpoints/grpo-iteration-20261028-01'
    for model in ('sft', 'previous', 'current'):
        rows = jsonl(current / f'{model}_predictions.jsonl')
        if len(rows) != 104 or len({r['case_id'] for r in rows}) != 104:
            raise ValueError('104 unique saved predictions required')
        expected = {**{k: 'train_diagnostic' for k in iteration['training_train_diagnostic_case_ids']},
                    **{k: 'development' for k in iteration['evaluation_development_case_ids']}}
        if {r['case_id']: r['suite'] for r in rows} != expected:
            raise ValueError('prediction suite differs')
        for row in rows:
            append(row, examples[row['case_id']], source='greedy_' + model, kind='prediction', suite=row['suite'],
                   index=0, role='evaluation_only', original=row)
    if len(result) != 4920:
        raise ValueError('all 4608 rollout and 312 saved prediction answers required')
    return result


def summarize_predictions(rows):
    result = {}
    for source, suite in sorted({(r['source'], r['suite']) for r in rows if r['kind'] == 'prediction'}):
        part = [r for r in rows if (r['source'], r['suite']) == (source, suite)]
        arithmetic = [r for r in part if r['graded_eligible']]
        result[f'{source}/{suite}'] = {'cases': len(part), 'arithmetic_cases': len(arithmetic),
            'original_strict_all_cases': sum(r['strict_reward'] for r in part) + sum(not r['graded_eligible'] and r['semantic_completed'] for r in part),
            'arithmetic_strict_full': sum(r['strict_reward'] for r in arithmetic),
            'arithmetic_verified_full': sum(r['verified_equivalent_reward'] for r in arithmetic),
            'arithmetic_graded_mean': sum(r['graded_reward'] for r in arithmetic) / len(arithmetic),
            'tiers': dict(Counter(r['graded_tier'] for r in arithmetic))}
    return result


def rescore(output):
    context = verified_inputs()
    plan, iteration, tok, cases, prompts, examples, files = context
    rows = gather_rows(context)
    groups, rollouts = summarize_rollouts(rows)
    # Reject accidentally relabelled case IDs, duplicated predictions, and changed files before saving.
    keys = [(r['source'], r['case_id'], r['sample_index']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate rescored answer identity')
    verify_files(ROOT, files)
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / 'rescored_answers.jsonl', rows)
    write_rows(output / 'group_diagnostics.jsonl', groups)
    write(output / 'rescore_manifest.json', {'plan': plan, 'input_and_source_sha256': files,
        'historical_files_unchanged': True, 'model_loaded': False, 'new_generation_calls': 0,
        'new_model_forward_calls': 0, 'optimizer_updates': 0, 'training_admitted': False})
    summary = {'contract_version': CONTRACT, 'records': len(rows), 'rollout_records': 4608, 'saved_prediction_records': 312,
        'levels': LEVELS, 'tier_counts_arithmetic': dict(Counter(r['graded_tier'] for r in rows if r['graded_eligible'])),
        'rollouts': rollouts, 'saved_predictions': summarize_predictions(rows),
        'new_generation_calls': 0, 'new_model_forward_calls': 0, 'optimizer_updates': 0, 'baseline_promoted': False,
        'training_admitted': False, 'strict_historical_rewards_unchanged': True,
        'files_sha256': {name: file_hash(output / name) for name in ('rescored_answers.jsonl', 'group_diagnostics.jsonl', 'rescore_manifest.json')}}
    write(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = rescore(args.output)
    print(json.dumps({'records': value['records'], 'rollout_records': value['rollout_records'],
                      'saved_prediction_records': value['saved_prediction_records'], 'tiers': value['tier_counts_arithmetic']}, indent=2))
