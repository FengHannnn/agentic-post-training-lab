"""Version bindings and explicit Adam lineage between two rollout policies."""

import hashlib
import json
from pathlib import Path


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_files(folder, files):
    folder = Path(folder).resolve()
    if not files:
        raise ValueError('nonempty file manifest required')
    for relative, expected in files.items():
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'missing, changed or unsafe file: {relative}')


def verify_parent(root, plan):
    parent = (Path(root) / plan['parent_run']).resolve()
    if not parent.is_relative_to(Path(root).resolve()):
        raise ValueError('unsafe parent path')
    if file_hash(parent / 'summary.json') != plan['parent_summary_sha256']:
        raise ValueError('parent summary identity differs')
    summary = json.loads((parent / 'summary.json').read_text(encoding='utf-8'))
    verify_files(parent, summary['files_sha256'])
    if (not summary['engineering_checks_passed'] or summary['logical_optimizer_steps'] != 16
            or summary['final_parameter_sha256'] != plan['parent_parameter_sha256']
            or summary['baseline_promoted']):
        raise ValueError('expected unpromoted first GRPO candidate required')
    for relative, key in [('final-checkpoint/adapter/adapter_model.safetensors', 'parent_adapter_sha256'),
                          ('final-checkpoint/optimizer.pt', 'parent_optimizer_sha256')]:
        if file_hash(parent / relative) != plan[key]:
            raise ValueError('parent checkpoint identity differs')
    manifest = json.loads((parent / 'run_manifest.json').read_text(encoding='utf-8'))
    return parent, summary, manifest


def seed_for(group, index):
    if not 0 <= group < 144 or not 0 <= index < 8:
        raise ValueError('invalid rollout group or sample index')
    return 202610280 + group * 1000 + index


def select_mixed_batches(rewards, case_ids):
    """All mixed groups exactly once in the ID order frozen before generation."""
    groups = rewards['groups']
    if len(groups) != len(case_ids) or {g['group_id'] for g in groups} != set(case_ids):
        raise ValueError('reward groups differ from frozen case IDs')
    selected = set()
    for g in groups:
        values = g['rewards']
        if len(values) != 8 or any(v not in (0., 1.) for v in values) or not g['eligible']:
            raise ValueError('eight eligible binary arithmetic rewards required')
        mixed = len(set(values)) == 2
        if g['zero_variance'] == mixed:
            raise ValueError('reward variance metadata differs')
        if mixed:
            selected.add(g['group_id'])
    ordered = [key for key in case_ids if key in selected]
    return [ordered[i:i + 4] for i in range(0, len(ordered), 4)]


def import_parent_adam(policy, optimizer, checkpoint, *, expected_hash, old_binding,
                       new_binding, expected_step=16):
    """Import verified moment buffers; do not reuse parent RNG or change its binding.

    The caller first checks the complete policy parameter digest against its parent.
    RNG for the new iteration is seeded independently; later within-round restores
    use the ordinary checkpoint restore function and the new input binding.
    """
    import torch

    checkpoint = Path(checkpoint)
    if old_binding == new_binding or file_hash(checkpoint) != expected_hash:
        raise ValueError('distinct bindings and verified parent optimizer required')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    state = saved['optimizer']
    if (saved['input_digest'] != old_binding or saved['step'] != expected_step
            or saved['trainable_names'] != [n for n, p in named]
            or not named or any('lora_' not in n for n, p in named)):
        raise ValueError('parent optimizer binding, step or parameter order differs')
    if len(state['param_groups']) != 1 or len(optimizer.param_groups) != 1:
        raise ValueError('single Adam parameter group required')
    group = state['param_groups'][0]
    if len(group['params']) != len(named) or len(set(group['params'])) != len(named):
        raise ValueError('parent optimizer parameter coverage differs')
    current = optimizer.state_dict()['param_groups'][0]
    # Device dispatch flags may vary with platform, mathematical hyperparameters may not.
    for key in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
        if group[key] != current[key]:
            raise ValueError(f'parent Adam hyperparameter differs: {key}')
    if set(state['state']) != set(group['params']):
        raise ValueError('parent Adam state coverage differs')
    for index, (_, parameter) in zip(group['params'], named):
        values = state['state'][index]
        if set(values) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('unexpected Adam state buffers')
        step = values['step']
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != expected_step:
            raise ValueError('parent Adam step counter differs')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = values[key]
            if (value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all()
                    or (key == 'exp_avg_sq' and (value < 0).any())):
                raise ValueError('invalid Adam moment shape, dtype or values')
    optimizer.load_state_dict(state)
    from agentic_post_training.trainers.grpo import compare_optimizer_snapshot
    comparison = compare_optimizer_snapshot(optimizer, state['state'], 0.)
    if not comparison['optimizer_state_passed']:
        raise ValueError('Adam import changed moments')
    return {'mode': 'carry_parent_adam', 'parent_optimizer_sha256': expected_hash,
            'parent_input_digest': old_binding, 'new_input_digest': new_binding,
            'parent_logical_steps': expected_step, 'adam_step_at_start': expected_step,
            'parent_rng_restored': False, **comparison}
