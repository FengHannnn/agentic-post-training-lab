"""Outcome GRPO groups, complete continuations and strict environment replay."""
import hashlib
import json
import math
from dataclasses import asdict
from .tasks import make_tasks
from .environment import Environment
from .runtime import SYSTEM, parse_action
from .dpo_recovery import build_context, SCENARIOS
from .rewards import training_samples


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def start_spec(index, seed, scenarios):
    if type(index) is not int or index < 0 or not scenarios or any(s not in SCENARIOS for s in scenarios):
        raise ValueError('valid group cursor and recovery scenarios required')
    tasks = [t for t in make_tasks() if t.split == 'train']
    task = tasks[(index // 2) % len(tasks)]
    scenario = scenarios[(index // 2) % len(scenarios)] if index % 2 else None
    return {'index': index, 'task_id': task.id, 'scenario': scenario,
            'kind': 'recovery' if scenario else 'complete', 'seed': seed + index * 100000}


def initial_state(task, scenario, tool_cost=.01):
    context = build_context(task, scenario, allowed_split=task.split) if scenario else None
    # Prefix observations keep the established recovery chat template exactly.
    env = Environment(task, 'potential', tool_cost=tool_cost)
    try:
        for item in (context['records'] if context else []):
            if env.step(parse_or_invalid(item['text'])) != item['row']:
                raise ValueError('recovery prefix replay differs')
        if env.done:
            raise ValueError('unfinished initial state required')
        env.reward_mode = 'outcome'
        messages = ([dict(m) for m in context['messages']] if context else
                    [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task.request()}])
        identity = {'task': asdict(task), 'scenario': scenario,
                    'prefix': context['records'] if context else [], 'tool_cost': tool_cost}
        return env, messages, digest(identity), identity
    except Exception:
        env.close()
        raise


def parse_or_invalid(text):
    try:
        return parse_action(text)
    except (ValueError, TypeError):
        return {'invalid_action': text}


def run_continuation(task, scenario, policy, sampling, version, seed, rollout_index, tool_cost=.01):
    env, messages, start_id, identity = initial_state(task, scenario, tool_cost)
    rows = []
    prefix_errors = env.tool_errors
    try:
        while not env.done:
            generation = policy(messages)
            row = env.step(parse_or_invalid(generation.text))
            row['generation'] = asdict(generation)
            row['sampling_seed'] = seed + len(rows)
            rows.append(row)
            messages.extend([{'role': 'assistant', 'content': generation.text},
                             {'role': 'user', 'content': json.dumps({'observation': row['observation']})}])
        return {'task_id': task.id, 'split': task.split, 'family_id': task.family_id,
                'scenario': scenario, 'start_id': start_id, 'start': identity,
                'sampling': sampling, 'policy_version': version, 'seed': seed,
                'rollout_index': rollout_index, 'status': 'complete', 'turns': rows,
                'success': env.success, 'termination': rows[-1]['termination'],
                'prefix_turns': len(identity['prefix']), 'total_turns': len(env.records),
                'return': sum(r['outcome_reward'] for r in env.records),
                'tool_errors': env.tool_errors - prefix_errors, 'total_tool_errors': env.tool_errors,
                'artifact': env.records[-1]['observation'].get('artifact') if env.success else None}
    except Exception as exc:
        exc.episode_evidence = {'task_id': task.id, 'scenario': scenario, 'start_id': start_id,
                                'policy_version': version, 'seed': seed, 'status': 'interrupted',
                                'training_admitted': False, 'turns': rows, 'error': str(exc)}
        raise
    finally:
        env.close()


def native_eos_token_ids(tokenizer):
    """Match vLLM 0.11 tokenizer EOS plus the frozen model generation-config EOS."""
    from pathlib import Path
    primary = tokenizer.eos_token_id
    if type(primary) is not int or primary < 0:
        raise ValueError('valid primary model EOS required')
    eos = {primary}
    folder = getattr(tokenizer, 'name_or_path', '')
    config = Path(folder)/'generation_config.json' if folder else None
    if config is not None and config.is_file():
        value = json.loads(config.read_text(encoding='utf8')).get('eos_token_id', primary)
        values = [value] if type(value) is int else value
        if (not isinstance(values, list) or not values
                or any(type(v) is not int or v < 0 for v in values)):
            raise ValueError('valid frozen model EOS token IDs required')
        eos.update(values)
    return eos


def validate_native(tokenizer, messages, generation, sampling):
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = generation.get('response_token_ids')
    probs = generation.get('behavior_token_log_probs')
    mode = 'sample' if sampling['temperature'] > 0 else 'greedy'
    if generation.get('mode') != mode or generation.get('prompt_token_ids') != list(prompt):
        raise ValueError('native prompt or generation mode differs')
    if not ids or any(type(x) is not int or x < 0 for x in list(prompt) + ids):
        raise ValueError('nonempty native tokens required')
    if len(prompt) + len(ids) > sampling['max_model_len'] or len(ids) > sampling['max_new_tokens']:
        raise ValueError('native token budget exceeded')
    if mode == 'sample' and (probs is None or len(probs) != len(ids)
                            or any(not math.isfinite(p) or p > 1e-6 for p in probs)):
        raise ValueError('finite aligned behavior probabilities required')
    if mode == 'greedy' and probs is not None:
        raise ValueError('greedy evaluation has no stochastic behavior q')
    if generation['raw_token_log_probs'] is not None:
        raise ValueError('processed collector must not relabel q as raw logp')
    if tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) != generation['text']:
        raise ValueError('native response decode differs')
    eos = native_eos_token_ids(tokenizer)
    if generation['finish_reason'] == 'stop':
        if ids[-1] not in eos or any(token in eos for token in ids[:-1]):
            raise ValueError('stop requires one native terminal EOS')
    elif generation['finish_reason'] == 'length':
        # vLLM checks the length cap before EOS; the last token may be EOS.
        if len(ids) != sampling['max_new_tokens'] or any(token in eos for token in ids[:-1]):
            raise ValueError('length action requires its actual sampled token limit')
    else:
        raise ValueError('unsupported inference termination')


def replay(task, episode, tokenizer=None):
    env, messages, start_id, identity = initial_state(task, episode['scenario'], episode['start']['tool_cost'])
    prefix_errors = env.tool_errors
    try:
        if episode['start_id'] != start_id or episode['start'] != identity:
            raise ValueError('episode initial state differs')
        if episode['status'] != 'complete' or not episode['turns']:
            raise ValueError('complete continuation required')
        for j, row in enumerate(episode['turns']):
            if env.done:
                raise ValueError('extra action after termination')
            g = row['generation']
            if tokenizer is not None:
                validate_native(tokenizer, messages, g, episode['sampling'])
            if row['sampling_seed'] != episode['seed'] + j:
                raise ValueError('action seed schedule differs')
            actual = env.step(parse_or_invalid(g['text']))
            if actual != {k: v for k, v in row.items() if k not in ('generation', 'sampling_seed')}:
                raise ValueError('environment action replay differs')
            messages.extend([{'role': 'assistant', 'content': g['text']},
                             {'role': 'user', 'content': json.dumps({'observation': actual['observation']})}])
        expected = {'task_id': task.id, 'split': task.split, 'family_id': task.family_id,
                    'success': env.success, 'total_turns': len(env.records),
                    'prefix_turns': len(identity['prefix']), 'termination': env.records[-1]['termination'],
                    'tool_errors': env.tool_errors - prefix_errors, 'total_tool_errors': env.tool_errors,
                    'artifact': env.records[-1]['observation'].get('artifact') if env.success else None}
        if not env.done or any(episode[k] != v for k, v in expected.items()):
            raise ValueError('complete episode summary differs')
        expected_return = sum(r['outcome_reward'] for r in env.records)
        if not math.isclose(episode['return'], expected_return, rel_tol=0, abs_tol=1e-12):
            raise ValueError('episode outcome return differs')
        return len(episode['turns'])
    finally:
        env.close()


def group_advantages(episodes, k=4, epsilon=1e-8):
    if type(k) is not int or k < 2 or len(episodes) != k or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('one complete K-rollout group and positive epsilon required')
    identity = None
    values = []
    for e in episodes:
        if e.get('status') != 'complete' or e.get('split') != 'train':
            raise ValueError('complete train episodes required')
        rows = e['turns']
        if not rows or not rows[-1]['done'] or any(r['done'] for r in rows[:-1]):
            raise ValueError('one complete environment termination required')
        current = (e['task_id'], e['start_id'], e['policy_version'], digest(e['sampling']))
        if identity is not None and identity != current:
            raise ValueError('same task, state, sampling and policy version required')
        identity = current
        if digest(e['start']) != e['start_id'] or e['start']['task']['id'] != e['task_id'] or e['start']['task']['split'] != 'train':
            raise ValueError('initial state binding differs')
        if e['total_turns'] != e['prefix_turns'] + len(rows) or e['prefix_turns'] != len(e['start']['prefix']):
            raise ValueError('episode action budget differs')
        expected = float(e['success']) - e['start']['tool_cost'] * e['total_turns']
        if not math.isfinite(e['return']) or not math.isclose(e['return'], expected, abs_tol=1e-10, rel_tol=0):
            raise ValueError('success minus full action cost required')
        if e['sampling']['temperature'] <= 0 or e['sampling'].get('logprobs_mode') != 'processed_logprobs':
            raise ValueError('processed stochastic sampling required')
        training_samples(e)
        values.append(e['return'])
    if len({e['seed'] for e in episodes}) != k or {e['rollout_index'] for e in episodes} != set(range(k)):
        raise ValueError('independent rollout seeds and indices required')
    mean = math.fsum(values) / k
    std = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / k)
    zero = max(values) == min(values)
    return {'returns': values, 'mean': mean, 'population_std': std, 'epsilon': epsilon,
            'zero_variance': zero, 'eligible': not zero,
            'advantages': [0.] * k if zero else [(v - mean) / (std + epsilon) for v in values]}
