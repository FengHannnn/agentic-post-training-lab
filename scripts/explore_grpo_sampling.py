"""Bounded 24-case temperature/sample-count probe; no training or promotion."""

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

from collect_vllm_rollout import ROOT, read, write, check_gpu, effective_config
from freeze_sft_baseline import verify as verify_baseline
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.rollouts.phase5_signal import run_seconds
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files

PLAN = ROOT / 'configs/experiments/grpo_sampling_exploration.json'
OUTPUT = ROOT / 'artifacts/trajectories/grpo-exploration-20261028-01'
ARCHIVE = ROOT / 'grpo-exploration-review-20261028.zip'
BUNDLE = ROOT / 'grpo_exploration_bundle_manifest.json'
STRATA = [('natural', 'left_plain'), ('natural', 'right_plain'), ('sequence', 'left_plain'),
          ('sequence', 'right_grouped'), ('natural', 'right_grouped'), ('symbolic', 'right_plain')]
EFFECTIVE = {'dtype': 'torch.float16', 'logprobs_mode': 'raw_logprobs', 'max_model_len': 512,
             'enforce_eager': True, 'enable_prefix_caching': False, 'enable_chunked_prefill': True,
             'max_num_seqs': 4, 'max_num_batched_tokens': 512, 'max_lora_rank': 8}


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def select_cases(train, old_rewards, new_rewards):
    """Choose four spread-out training families shared by four persistent-zero strata."""
    arithmetic = [r for r in train if r.expected_type == 'tool_call']
    indices = []
    for audit in (old_rewards, new_rewards):
        groups = audit['groups']
        indexed = {g['group_id']: g for g in groups}
        if len(indexed) != len(groups) or set(indexed) != {r.id for r in arithmetic}:
            raise ValueError('historical rewards must cover all arithmetic training inputs')
        for group in groups:
            if (not group['eligible'] or len(group['rewards']) != 8
                    or any(v not in (0., 1.) for v in group['rewards'])):
                raise ValueError('eligible eight-answer binary historical groups required')
        indices.append(indexed)
    common = []
    for wording, category in STRATA[:4]:
        common.append({r.family_id for r in arithmetic
                       if r.provenance['wording'] == wording and r.category == category
                       and all(sum(index[r.id]['rewards']) == 0 for index in indices)})
    families = sorted(set.intersection(*common))
    if len(families) < 4:
        raise ValueError('four shared historical-zero training families required')
    selected_families = [families[round(i * (len(families) - 1) / 3)] for i in range(4)]
    result = []
    for family in selected_families:
        for wording, category in STRATA:
            matches = [r for r in arithmetic if r.family_id == family
                       and r.provenance['wording'] == wording and r.category == category]
            if len(matches) != 1:
                raise ValueError('one unique training input per family/stratum required')
            result.append(matches[0])
    return result, selected_families


def seed_for(group, index):
    if not 0 <= group < 24 or not 0 <= index < 32:
        raise ValueError('invalid probe group/sample index')
    return 2026102810 + group * 1000 + index


def requests():
    """Rotate temperature order across four blocks; each arm keeps the same sample seeds."""
    arms = ['t100', 't130', 't160']
    result = []
    for block in range(4):
        offset = block % 3
        for arm in arms[offset:] + arms[:offset]:
            for group in range(24):
                for index in range(block * 8, (block + 1) * 8):
                    result.append((arm, group, index, seed_for(group, index)))
    return result


def sampling(eos, temperature):
    if temperature not in (1., 1.3, 1.6):
        raise ValueError('unplanned temperature')
    return dict(n=1, temperature=temperature, top_p=1, top_k=-1, max_tokens=64, min_tokens=0,
                repetition_penalty=1, presence_penalty=0, frequency_penalty=0, logprobs=1,
                stop_token_ids=[eos], ignore_eos=False, detokenize=False)


def setup():
    from transformers import AutoTokenizer
    plan = read(PLAN)
    fixed = {'contract_version': 'grpo-sampling-exploration-v1', 'samples_per_case_arm': 32,
             'sample_prefixes': [8, 16, 32], 'seed_base': 2026102810, 'seed_group_stride': 1000,
             'temperatures': {'t100': 1., 't130': 1.3, 't160': 1.6}, 'chunk_requests': 64,
             'max_wall_seconds': 1800, 'max_run_cost_yuan': 10., 'optimizer_updates': 0,
             'auto_promote_baseline': False, 'reward_contract': 'arithmetic-strict-v1'}
    if any(plan.get(key) != value for key, value in fixed.items()):
        raise ValueError('fixed sampling-exploration contract differs')
    candidate = ROOT / plan['candidate_run']
    if file_hash(candidate / 'summary.json') != plan['candidate_summary_sha256']:
        raise ValueError('candidate summary identity differs')
    summary = read(candidate / 'summary.json')
    verify_files(candidate, summary['files_sha256'])
    candidate_manifest = read(candidate / 'run_manifest.json')
    verify_files(candidate / 'source', candidate_manifest['source_sha256'])
    verify_files(ROOT, candidate_manifest['source_sha256'])
    if (not summary['engineering_checks_passed'] or summary['cumulative_candidate_steps'] != 31
            or summary['final_parameter_sha256'] != plan['candidate_parameter_sha256']
            or summary['baseline_promoted'] or candidate_manifest['device'] != 'NVIDIA L20'
            or file_hash(candidate / 'final-checkpoint/adapter/adapter_model.safetensors') != plan['candidate_adapter_sha256']):
        raise ValueError('audited unpromoted 31-step candidate required')
    baseline = ROOT / 'artifacts/baselines/sft-v2-phase3'
    if file_hash(baseline / 'manifest.json') != plan['baseline_manifest_sha256']:
        raise ValueError('baseline identity differs')
    verify_baseline(baseline)
    manifest = read(baseline / 'manifest.json')
    base = (ROOT / manifest['base_cache_files'][0]['source']).parent
    folder = ROOT / 'data/preferences/dpo_v2'
    if file_hash(folder / 'manifest.json') != candidate_manifest['data_manifest_sha256']:
        raise ValueError('candidate preference dataset identity differs')
    data_manifest = read(folder / 'manifest.json')
    splits = load_preference_splits(folder, data_manifest)
    historical = []
    for item in plan['historical_rollouts']:
        source = ROOT / item['path']
        if file_hash(source / 'summary.json') != item['summary_sha256']:
            raise ValueError('historical selection input changed')
        verify_files(source, read(source / 'summary.json')['files_sha256'])
        historical.append(read(source / 'reward_audit.json'))
    selected, families = select_cases(splits['train'], *historical)
    if [r.id for r in selected] != plan['case_ids'] or families != plan['family_ids']:
        raise ValueError('precommitted selected cases/families differ')
    if len(selected) != 24 or len({r.id for r in selected}) != 24:
        raise ValueError('24 unique training cases required')
    if ({r.family_id for r in selected} & {r.family_id for r in splits['development']}
            or {r.id for r in selected} & {r.id for r in splits['development']}
            or {(r.system, r.prompt) for r in selected} & {(r.system, r.prompt) for r in splits['development']}):
        raise ValueError('development input leaked into sampling')
    tok = AutoTokenizer.from_pretrained(str(baseline / 'checkpoint/tokenizer'), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cases = [r.target() for r in selected]
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': c.system}, {'role': 'user', 'content': c.prompt}],
                                      tokenize=True, add_generation_prompt=True) for c in cases]
    if any(len(prompt) + 64 > 512 for prompt in prompts):
        raise ValueError('context budget exceeded')
    config = read(ROOT / 'configs/rollout_vllm_smoke.json')
    return plan, config, candidate, baseline, base, tok, selected, cases, prompts


def identity(context):
    plan, config, candidate, baseline, base, tok, selected, cases, prompts = context
    sources = read(candidate / 'run_manifest.json')['source_sha256'].copy()
    for path in (Path(__file__), PLAN):
        sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return {'plan': plan, 'config': config, 'source_sha256': sources,
            'baseline_manifest_sha256': file_hash(baseline / 'manifest.json'),
            'candidate_summary_sha256': file_hash(candidate / 'summary.json'),
            'candidate_adapter_sha256': file_hash(candidate / 'final-checkpoint/adapter/adapter_model.safetensors'),
            'candidate_parameter_sha256': plan['candidate_parameter_sha256'],
            'data_manifest_sha256': file_hash(ROOT / 'data/preferences/dpo_v2/manifest.json'),
            'development_inputs_sampled': False, 'training_admitted': False, 'optimizer_updates': 0,
            'probability_semantics': 'raw model logp before temperature; behavior equals raw only for t100; behavior not recorded for hotter arms'}


def native_record(output, prompt, case, arm, index, seed, temperature, tok):
    if list(output.prompt_token_ids) != prompt or len(output.outputs) != 1:
        raise ValueError('native prompt or completion count differs')
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    if (not tokens or len(tokens) > 64 or any(type(t) is not int or t < 0 for t in tokens)
            or completion.logprobs is None or len(completion.logprobs) != len(tokens)
            or any(t not in p for t, p in zip(tokens, completion.logprobs))):
        raise ValueError('invalid native token/probability alignment')
    raw = [p[t].logprob for t, p in zip(tokens, completion.logprobs)]
    if any(not math.isfinite(v) or v > 1e-6 for v in raw) or tok.eos_token_id in tokens[:-1]:
        raise ValueError('invalid raw logp or tokens after EOS')
    if completion.finish_reason == 'stop' and tokens[-1] == tok.eos_token_id:
        finish = 'eos'
    elif completion.finish_reason == 'length' and len(tokens) == 64 and tokens[-1] != tok.eos_token_id:
        finish = 'length'
    else:
        raise ValueError('unsupported native finish reason')
    text = tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {'arm': arm, 'temperature': temperature, 'case': asdict(case), 'case_id': case.id,
            'sample_index': index, 'seed': seed, 'prompt_token_ids': list(prompt), 'response_token_ids': tokens,
            'response_mask': [True] * len(tokens), 'sampler_raw_model_token_log_probs': raw,
            'behavior_token_log_probs': list(raw) if temperature == 1. else None,
            'eos_token_id': tok.eos_token_id, 'finish_reason': finish, 'text': text,
            'raw_text': tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            'training_admitted': False, **score_reward(text, case, finish_reason=finish)}


def validate_records(rows, plan, cases, prompts, tok):
    schedule = requests()
    if len(rows) != len(schedule):
        raise ValueError('all 2304 planned requests must be retained')
    for row, (arm, group, index, seed) in zip(rows, schedule):
        case = cases[group]
        tokens = row['response_token_ids']
        raw = row['sampler_raw_model_token_log_probs']
        if (row['arm'] != arm or row['temperature'] != plan['temperatures'][arm]
                or row['case_id'] != case.id or row['case'] != asdict(case)
                or row['sample_index'] != index or row['seed'] != seed or row['prompt_token_ids'] != prompts[group]
                or not tokens or len(tokens) > 64 or any(type(t) is not int or t < 0 for t in tokens)
                or len(raw) != len(tokens) or any(not math.isfinite(v) or v > 1e-6 for v in raw)
                or row['response_mask'] != [True] * len(tokens) or row['eos_token_id'] != tok.eos_token_id
                or tok.eos_token_id in tokens[:-1] or row['finish_reason'] not in ('eos', 'length')
                or (tokens[-1] == tok.eos_token_id) != (row['finish_reason'] == 'eos')
                or (row['finish_reason'] == 'length' and len(tokens) != 64)
                or row['behavior_token_log_probs'] != (raw if row['temperature'] == 1. else None)
                or row['text'] != tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                or row['raw_text'] != tok.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                or row['training_admitted']):
            raise ValueError('native probe identity, mask, raw/behavior semantics or decode differs')
        reward = score_reward(row['text'], case, finish_reason=row['finish_reason'])
        if any(row[k] != value for k, value in reward.items()):
            raise ValueError('saved strict reward/semantic scoring differs')


def summarize(rows, selected, plan):
    """Descriptive, equal-budget arm comparison and nested sample-count comparison."""
    indexed = {(r['arm'], r['case_id'], r['sample_index']): r for r in rows}
    if len(indexed) != len(rows) or len(rows) != 2304:
        raise ValueError('duplicate or incomplete exploration records')
    results = {}
    for arm in plan['temperatures']:
        arm_results = {}
        for n in plan['sample_prefixes']:
            per_case = []
            for example in selected:
                samples = [indexed[arm, example.id, i] for i in range(n)]
                passed = [i for i, r in enumerate(samples) if r['reward'] == 1.]
                role = 'persistent_zero' if (example.provenance['wording'], example.category) in STRATA[:4] else 'near_control' if example.provenance['wording'] == 'natural' else 'easy_control'
                per_case.append({'case_id': example.id, 'family_id': example.family_id,
                    'wording': example.provenance['wording'], 'category': example.category, 'role': role,
                    'samples': n, 'strict_successes': len(passed), 'has_strict_success': bool(passed),
                    'mixed': 0 < len(passed) < n, 'first_strict_success_at': passed[0] + 1 if passed else None,
                    'semantic_completed': sum(r['semantic_pass'] and r['finish_reason'] == 'eos' for r in samples),
                    'semantic_only_completed': sum(r['semantic_pass'] and r['reward'] == 0 and r['finish_reason'] == 'eos' for r in samples),
                    'invalid_json': sum(not r['score']['json_valid'] for r in samples),
                    'length': sum(r['finish_reason'] == 'length' for r in samples),
                    'unique_native_answers': len({tuple(r['response_token_ids']) for r in samples})})
            def aggregate(values):
                return {'cases': len(values), 'samples': n * len(values),
                    'strict_successes': sum(v['strict_successes'] for v in values),
                    'cases_with_strict_success': sum(v['has_strict_success'] for v in values),
                    'mixed_groups': sum(v['mixed'] for v in values),
                    'nonzero_advantage_completions_if_grouped': n * sum(v['mixed'] for v in values),
                    'strict_success_per_1000_samples': 1000 * sum(v['strict_successes'] for v in values) / (n * len(values)),
                    **{key: sum(v[key] for v in values) for key in ('semantic_completed', 'semantic_only_completed', 'invalid_json', 'length')}}
            arm_results[str(n)] = {'overall': aggregate(per_case),
                'by_role': {role: aggregate([v for v in per_case if v['role'] == role])
                            for role in ('persistent_zero', 'near_control', 'easy_control')},
                'by_stratum': {f'{wording}/{category}': aggregate([v for v in per_case if v['wording'] == wording and v['category'] == category])
                               for wording, category in STRATA}, 'per_case': per_case}
        results[arm] = arm_results
    base = results['t100']
    newly_reached = [r['case_id'] for r, first in zip(base['32']['per_case'], base['8']['per_case'])
                     if r['role'] == 'persistent_zero' and r['has_strict_success'] and not first['has_strict_success']]
    comparisons = {}
    for arm in ('t130', 't160'):
        left, right = base['32'], results[arm]['32']
        comparisons[arm] = {'primary_newly_reached_case_ids': [b['case_id'] for a, b in zip(left['per_case'], right['per_case'])
            if b['role'] == 'persistent_zero' and not a['has_strict_success'] and b['has_strict_success']],
            'primary_lost_case_ids': [a['case_id'] for a, b in zip(left['per_case'], right['per_case'])
            if a['role'] == 'persistent_zero' and a['has_strict_success'] and not b['has_strict_success']],
            'primary_mixed_group_delta': right['by_role']['persistent_zero']['mixed_groups'] - left['by_role']['persistent_zero']['mixed_groups'],
            'overall_invalid_json_delta': right['overall']['invalid_json'] - left['overall']['invalid_json'],
            'overall_length_delta': right['overall']['length'] - left['overall']['length']}
    return {'by_arm_and_prefix': results, 'count_effect_t100_prefix32_vs8': {'primary_newly_reached_case_ids': newly_reached},
            'temperature_equal_budget_32': comparisons, 'automatic_training_or_temperature_selection': False}


def preflight(output=None):
    context = setup()
    plan, config, candidate, baseline, base, tok, selected, cases, prompts = context
    seeds = {seed_for(g, i) for g in range(24) for i in range(32)}
    old_seeds = {20261021 + g * 1000 + i for g in range(144) for i in range(8)}
    old_seeds |= {202610280 + g * 1000 + i for g in range(144) for i in range(8)}
    if len(seeds) != 768 or seeds & old_seeds:
        raise ValueError('seed collision with past sampling')
    result = {'contract_version': plan['contract_version'], 'input_preflight_passed': True,
              'candidate_parameter_sha256': plan['candidate_parameter_sha256'],
              'case_ids': plan['case_ids'], 'family_ids': plan['family_ids'], 'cases': 24,
              'primary_historical_zero_cases': 16, 'controls': 8, 'new_completions_planned': len(requests()),
              'max_prompt_tokens': max(map(len, prompts)), 'prefixes': [8, 16, 32],
              'temperatures': plan['temperatures'], 'seed_unique_within_each_arm': True,
              'same_case_index_seed_shared_across_temperatures': True, 'gpu_runtime_checked': False,
              'model_forward_calls': 0, 'optimizer_updates': 0, 'plan_sha256': file_hash(PLAN)}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        write(output, result)
    return result


def check_device(config):
    device = check_gpu(config)
    if device != 'NVIDIA L20':
        raise ValueError('this probe is pinned to NVIDIA L20')
    return device


def load_run(output):
    context = setup()
    inv = read(output / 'run_manifest.json')
    if any(inv[k] != value for k, value in identity(context).items()) or inv['device'] != 'NVIDIA L20':
        raise ValueError('sampling probe identity differs')
    verify_files(output / 'source', inv['source_sha256'])
    return context


def model_check(output):
    import torch
    from train_grpo_first import load_model
    from agentic_post_training.trainers.dpo import parameter_digest
    context = setup()
    plan, config, candidate, baseline, base, tok, selected, cases, prompts = context
    output.mkdir(parents=True, exist_ok=False)
    inv = {**identity(context), 'device': check_device(config)}
    write(output / 'run_manifest.json', inv)
    for relative, digest in inv['source_sha256'].items():
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if file_hash(target) != digest:
            raise ValueError('source changed during snapshot')
    model = load_model(base, candidate / 'final-checkpoint/adapter', False)
    actual = parameter_digest(model)
    passed = actual == plan['candidate_parameter_sha256'] and all(not p.requires_grad and p.grad is None for p in model.parameters())
    write(output / 'model_check.json', {'passed': passed, 'parameter_sha256': actual,
          'all_parameters_frozen': all(not p.requires_grad for p in model.parameters()),
          'gradients_absent': all(p.grad is None for p in model.parameters()),
          'model_forward_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0})
    if not passed:
        raise ValueError('frozen 31-step candidate reconstruction failed')


def generate(output):
    context = load_run(output)
    plan, config, candidate, baseline, base, tok, selected, cases, prompts = context
    if not read(output / 'model_check.json')['passed']:
        raise ValueError('model check must pass before generation')
    check_device(config)
    if (output / 'rollouts.jsonl').exists() or (output / 'generation_checks.json').exists():
        raise FileExistsError('fresh generation outputs required')
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import torch
    import torch.distributed as dist
    started = time.perf_counter()
    engine = LLM(model=str(base), tokenizer=str(baseline / 'checkpoint/tokenizer'), dtype='half',
        trust_remote_code=False, enable_lora=True, max_lora_rank=8, max_loras=1, max_model_len=512,
        max_num_seqs=4, max_num_batched_tokens=512, gpu_memory_utilization=.25, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True, seed=plan['seed_base'],
        generation_config='vllm', logprobs_mode='raw_logprobs')
    load_seconds = time.perf_counter() - started
    effective = effective_config(engine)
    counts = Counter()
    total_tokens = 0
    schedule = requests()
    request = LoRARequest('grpo-31step-sampling-probe', 1, str(candidate / 'final-checkpoint/adapter'))
    try:
        if effective != EFFECTIVE:
            raise ValueError('effective native engine settings differ')
        torch.cuda.synchronize()
        started = time.perf_counter()
        with (output / 'rollouts.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for start in range(0, len(schedule), 64):
                chunk = schedule[start:start + 64]
                params = [SamplingParams(seed=seed, **sampling(tok.eos_token_id, plan['temperatures'][arm]))
                          for arm, group, index, seed in chunk]
                native = engine.generate([{'prompt_token_ids': prompts[g]} for arm, g, i, seed in chunk],
                                         params, lora_request=request, use_tqdm=False)
                if len(native) != len(chunk):
                    raise ValueError('missing native completions')
                for result, (arm, group, index, seed) in zip(native, chunk):
                    row = native_record(result, prompts[group], cases[group], arm, index, seed,
                                        plan['temperatures'][arm], tok)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    total_tokens += len(row['response_token_ids'])
                    counts[arm] += 1
                stream.flush()
                print(f'Sampling probe saved {start + len(chunk)}/2304', flush=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized():
            dist.destroy_process_group()
    write(output / 'generation_checks.json', {'effective_engine': effective, 'engine_load_seconds': load_seconds,
          'generation_seconds': seconds, 'generation_requests': len(schedule), 'engine_generate_calls': 36,
          'requests_per_arm': dict(counts), 'response_tokens': total_tokens, 'engine_shutdown_called': True,
          'sampling_parameters': {arm: sampling(tok.eos_token_id, value) for arm, value in plan['temperatures'].items()}})


def audit(output, *, write_summary=True):
    context = load_run(output)
    plan, config, candidate, baseline, base, tok, selected, cases, prompts = context
    rows = jsonl(output / 'rollouts.jsonl')
    validate_records(rows, plan, cases, prompts, tok)
    model = read(output / 'model_check.json')
    if (model['parameter_sha256'] != plan['candidate_parameter_sha256'] or not model['passed']
            or not model['all_parameters_frozen'] or not model['gradients_absent']
            or model['model_forward_calls'] != 0 or model['backward_calls'] != 0 or model['optimizer_updates'] != 0):
        raise ValueError('frozen model identity record differs')
    gen = read(output / 'generation_checks.json')
    if (gen['effective_engine'] != EFFECTIVE or gen['generation_requests'] != 2304 or gen['engine_generate_calls'] != 36
            or gen['requests_per_arm'] != {arm: 768 for arm in plan['temperatures']}
            or gen['sampling_parameters'] != {arm: sampling(tok.eos_token_id, value) for arm, value in plan['temperatures'].items()}
            or gen['response_tokens'] != sum(len(r['response_token_ids']) for r in rows)
            or not gen['engine_shutdown_called'] or not math.isfinite(gen['generation_seconds']) or gen['generation_seconds'] <= 0):
        raise ValueError('generation configuration/counts differ')
    verify_files(ROOT, read(output / 'run_manifest.json')['source_sha256'])
    result = {'contract_version': plan['contract_version'], 'exploration_completed': True,
              'native_record_audit_passed': True, 'candidate_parameter_sha256': plan['candidate_parameter_sha256'],
              'cases': 24, 'samples': 2304, 'finish_reasons': dict(Counter(r['finish_reason'] for r in rows)),
              'response_tokens': gen['response_tokens'], 'generation_seconds': gen['generation_seconds'],
              'backward_calls': 0, 'optimizer_updates': 0, 'training_admitted': False, 'baseline_promoted': False,
              'behavior_probability_recorded_only_for_t100': True, **summarize(rows, selected, plan),
              'files_sha256': {name: file_hash(output / name) for name in (
                  'run_manifest.json', 'model_check.json', 'generation_checks.json', 'rollouts.jsonl')}}
    if write_summary:
        write(output / 'summary.json', result)
    elif read(output / 'summary.json') != result:
        raise ValueError('exploration summary differs from recomputed records')
    return result


def run(args):
    price = args.hourly_price if args.hourly_price is not None else float(input('GPU hourly price (yuan): '))
    seconds = run_seconds(price)
    bundle = read(BUNDLE)
    verify_files(ROOT, bundle['files'])
    verify_files(ROOT, bundle['required_existing_files'])
    if args.output.exists() or ARCHIVE.exists():
        raise FileExistsError('fresh exploration output/archive required; no automatic retry')
    preflight()
    started = time.perf_counter()
    stages = []
    error = None
    try:
        for stage in ('model-check', 'generate', 'audit'):
            remaining = seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError('bounded sampling job expired')
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                                     '--output', str(args.output.resolve())], cwd=ROOT, start_new_session=(os.name != 'nt'))
            try:
                code = proc.wait(timeout=remaining)
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
                raise TimeoutError('sampling stage stopped; no automatic retry')
            stages.append({'stage': stage, 'exit_code': code})
            if code:
                raise RuntimeError(f'{stage} exited {code}; later stages not started')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        elapsed = time.perf_counter() - started
        write(args.output / 'cost_and_run.json', {'elapsed_run_seconds': elapsed, 'hourly_price_yuan': price,
              'estimated_run_cost_yuan': elapsed * price / 3600, 'error': error, 'stages': stages,
              'max_run_seconds': seconds, 'max_run_cost_yuan': 10., 'automatic_retry': False,
              'instance_automatically_shutdown': False, 'billing_scope': 'bounded stages only; preflight/packing/boot/idle/storage/rounding excluded'})
        with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(args.output.parent).as_posix())
        print(f'Download from project root: {ARCHIVE}', flush=True)
    if error:
        raise RuntimeError(error)


def main():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_NO_USAGE_STATS='1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('run', 'preflight', 'model-check', 'generate', 'audit'), default='run')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--hourly-price', type=float)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    if args.stage == 'run':
        run(args)
    elif args.stage == 'preflight':
        print(json.dumps(preflight(args.preflight_output), ensure_ascii=False), flush=True)
    elif args.stage == 'model-check':
        model_check(args.output)
    elif args.stage == 'generate':
        generate(args.output)
    else:
        result = audit(args.output)
        print(json.dumps({key: result[key] for key in ('exploration_completed', 'samples', 'finish_reasons',
              'count_effect_t100_prefix32_vs8', 'temperature_equal_budget_32', 'optimizer_updates')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
