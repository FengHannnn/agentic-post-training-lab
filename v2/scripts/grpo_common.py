"""Frozen C2 inputs, transaction IO, runtime binding and resumable checkpoint identity."""
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import ROOT, file_hash, verify_files
from apt_v2.grpo import digest, start_spec


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def jsonl(path, row):
    with Path(path).open('a', encoding='utf8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def under(root, name):
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('path outside run root')
    return path


def validate_config(cfg):
    from apt_v2.grpo_repair_study import expected_source_sft_updates
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{5,100}', cfg['run_id']):
        raise ValueError('safe unique run ID required')
    if (cfg['algorithm'] not in ('outcome_grpo','fine_grained_grpo') or cfg['phase'] not in ('preflight', 'formal')
            or cfg['source_sft_updates'] != expected_source_sft_updates(cfg) or cfg['source_dpo_updates'] != 11
            or cfg['updates_per_group'] != 1 or cfg['k'] < 2 or cfg['tool_cost'] != .01
            or cfg['loss_reduction'] != 'token_mean_then_action_mean_then_episode_mean'
            or cfg['zero_variance'] != 'skip_without_rejection_resampling'):
        raise ValueError('C2 outcome contract differs')
    method=cfg.get('credit_method','outcome')
    if ((cfg['algorithm']=='fine_grained_grpo') != (method in ('potential_rtg_depth_v1','outcome_local_verified_v2'))
            or method not in ('outcome','potential_rtg_depth_v1','outcome_local_verified_v2')):
        raise ValueError('algorithm and action-credit method differ')
    from apt_v2.dpo_recovery import SCENARIOS
    if cfg.get('curriculum_schema')=='repair_study_v1':
        from apt_v2.grpo_repair_study import validate_config as validate_repair
        validate_repair(cfg)
    elif cfg.get('curriculum_schema')=='alias_curriculum_v1':
        from apt_v2.grpo_alias_curriculum import validate_config as validate_alias
        validate_alias(cfg)
    elif cfg.get('curriculum_schema')=='fine_credit_ablation_v1':
        from apt_v2.grpo_fine_study import validate_config as validate_fine
        validate_fine(cfg)
    elif cfg.get('curriculum_schema'):
        from apt_v2.grpo_curriculum import validate_config as validate_curriculum
        validate_curriculum(cfg)
    elif set(cfg['recovery_scenarios']) != set(SCENARIOS) or len(cfg['recovery_scenarios']) != len(SCENARIOS):
        raise ValueError('seven distinct train recovery scenarios required')
    if method=='outcome_local_verified_v2' and cfg.get('curriculum_schema') not in ('fine_credit_ablation_v1','alias_curriculum_v1','repair_study_v1'):
        raise ValueError('local verified credit requires the frozen fine study')
    limits = (cfg['max_groups'], cfg['max_updates'], cfg['k'], cfg['max_wall_seconds'])
    if any(type(v) is not int or v <= 0 for v in limits) or cfg['max_updates'] > cfg['max_groups']:
        raise ValueError('positive finite group/update/time limits required')
    wall_cap = cfg['max_wall_seconds'] if cfg.get('curriculum_schema') else (3600 if cfg['phase'] == 'preflight' else 14400)
    if cfg['max_wall_seconds'] > wall_cap:
        raise ValueError('authorized wall-time cap exceeded')
    prior = cfg.get('prior_preflight_runs', [])
    if any(not math.isfinite(r['elapsed_seconds']) or r['elapsed_seconds'] < 0 for r in prior):
        raise ValueError('finite prior preflight elapsed times required')
    if cfg['phase'] == 'preflight' and prior and math.fsum(r['elapsed_seconds'] for r in prior) + cfg['max_wall_seconds'] > 3600:
        raise ValueError('combined preflight wall-time budget exceeded')
    if cfg['phase'] == 'preflight' and (cfg['evaluation']['enabled'] or not 1 <= cfg['minimum_preflight_updates'] <= cfg['max_updates']):
        raise ValueError('bounded real-update preflight required')
    if cfg['phase'] == 'formal':
        cursors = cfg['group_evaluation_cursors']
        if (not cfg['evaluation']['enabled'] or cfg['max_groups'] % 2 or cfg['max_updates'] != cfg['max_groups']
                or not cursors or cursors != sorted(set(cursors)) or cursors[-1] != cfg['max_groups']
                or any(type(v) is not int or v <= 0 or v % 2 for v in cursors)
                or not 1 <= cfg['formal_job_index'] <= 3 or cfg['minimum_actual_updates'] < (0 if cfg.get('curriculum_schema') in ('alias_curriculum_v1','repair_study_v1') else 1)):
            raise ValueError('fixed balanced formal group budget and evaluation cursors required')
        if (cfg['eval']['max_model_len'] != cfg['sampling']['max_model_len']
                or cfg['eval']['max_new_tokens'] != cfg['sampling']['max_new_tokens']
                or cfg['sample_eval']['rollouts_per_task'] != 4 or cfg['recovery_eval']['samples_per_context'] != 3
                or cfg['protocol']['quality_contexts_per_model'] != 7 or cfg['protocol']['quality_repeats'] != 2):
            raise ValueError('fixed 132-episode paired evaluation and 7x2 quality protocol required')
    s = cfg['sampling']
    if (s['temperature'] != 1 or s['top_p'] != 1 or s['top_k'] != -1 or s['min_p'] != 0
            or s['min_tokens'] != 0 or s['repetition_penalty'] != 1
            or s['presence_penalty'] != 0 or s['frequency_penalty'] != 0
            or s['logprobs_mode'] != 'processed_logprobs'):
        raise ValueError('first collector requires untruncated temperature-one behavior sampling')
    if not 0 < s['max_new_tokens'] < s['max_model_len']:
        raise ValueError('response and context budget differs')
    under(ROOT, cfg['initial_dir'])
    under(ROOT, cfg['base_dir'])


def runtime_identity():
    import importlib.metadata
    import platform
    import sqlite3
    import torch
    if not torch.cuda.is_available():
        raise ValueError('CUDA device required for real GRPO preflight')
    p = torch.cuda.get_device_properties(0)
    return {'gpu_name': p.name, 'gpu_uuid': str(p.uuid), 'gpu_total_memory': p.total_memory,
            'compute_capability': list(torch.cuda.get_device_capability(0)),
            'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'sqlite': sqlite3.sqlite_version,
            'packages': {n: importlib.metadata.version(n) for n in ('vllm', 'transformers', 'peft')}}


def validate_runtime(cfg, actual, expected=None):
    if cfg.get('required_gpu_name') and actual['gpu_name'] != cfg['required_gpu_name']:
        raise ValueError('configured GPU class differs')
    if actual['gpu_total_memory'] < cfg.get('required_min_gpu_bytes', 0):
        raise ValueError('configured minimum GPU capacity required')
    if actual['packages'] != cfg['required_packages']:
        raise ValueError('configured software versions differ: '+str(actual['packages']))
    if expected is not None and actual != expected:
        raise ValueError('within-job GPU/software identity differs')


def checkpoint_binding(folder):
    return {p.relative_to(folder).as_posix(): file_hash(p) for p in sorted(Path(folder).rglob('*'))
            if p.is_file() and p.name != 'state.pt' and '.writing' not in p.name}


def load_transition(output, index, before):
    path = output / 'transitions' / f'group-{index:04}.json'
    result = read(path)
    cfg = read(output / 'config.json')
    if result['index'] != index or result['before'] != before or result['config_sha256'] != file_hash(output / 'config.json'):
        raise ValueError('transition checkpoint or config differs')
    gp = output / 'groups' / f'group-{index:04}.json'
    if result['group_sha256'] != file_hash(gp):
        raise ValueError('transition rollout binding differs')
    if result['updated']:
        checkpoint = under(output, result['after']['checkpoint'])
        verify_files(checkpoint, read(checkpoint / 'complete.json')['files'])
        complete = read(checkpoint / 'complete.json')
        if (complete['policy_version'] != result['after']['version']
                or complete['actual_updates'] != result['after']['updates']
                or file_hash(checkpoint / 'adapter/adapter_model.safetensors') != result['after']['adapter_sha256']
                or complete['config_sha256'] != file_hash(output / 'config.json')
                or complete['group_cursor'] != index + 1):
            raise ValueError('committed checkpoint ancestry differs')
    elif result['after'] != before:
        raise ValueError('skipped update changed policy state')
    return result


def initial_policy(cfg):
    return {'checkpoint': 'initial', 'version': 0, 'updates': 0, 'adapter_sha256': cfg['adapter_sha256']}


def verify_inputs(output, runtime=True, base_weights=False, local=False):
    cfg = read(output / 'config.json')
    validate_config(cfg)
    verify_files(ROOT, read(output / 'run_manifest.json')['source_sha256'])
    binding = read(output / 'input_binding.json')
    verify_files(output, binding['files'])
    if binding['config_sha256'] != file_hash(output / 'config.json'):
        raise ValueError('frozen config differs')
    model_files = read(output / 'initial/model_identity.json')['files_sha256']
    if not local:
        verify_files(under(ROOT, cfg['base_dir']), model_files if base_weights else
                     {n: h for n, h in model_files.items() if not n.endswith('.safetensors')})
    if runtime:
        validate_runtime(cfg, runtime_identity(), read(output / 'execution_environment.json'))
    return cfg


def prepare(output):
    import shutil
    from tqdm.auto import tqdm
    cfg = read(output / 'config.json')
    validate_config(cfg)
    actual = runtime_identity()
    atomic_json(output / 'execution_environment.json', actual)
    validate_runtime(cfg, actual)
    initial = under(ROOT, cfg['initial_dir'])
    complete = read(initial / 'complete.json')
    verify_files(initial / 'adapter', complete['adapter_sha256'])
    if (complete['total_sft_updates'] != cfg['source_sft_updates'] or complete['total_dpo_updates'] != 11
            or file_hash(initial / 'adapter/adapter_model.safetensors') != cfg['adapter_sha256']):
        raise ValueError('reviewed W identity differs')
    if cfg.get('curriculum_schema')=='repair_study_v1':
        source_audit=read(initial/'source_audit.json')
        if not source_audit.get('passed'):raise ValueError('repair initial source audit failed')
        from apt_v2.grpo_repair_study import warm_recipe_counts
        expected_steps,expected_exposures=warm_recipe_counts(cfg)
        if cfg['repair_arm']=='warm' and (complete.get('parent_W_sha256')!=cfg['baseline_W_sha256']
                or source_audit.get('actual_updates')!=expected_steps
                or source_audit.get('action_exposures')!=expected_exposures):
            raise ValueError('B warm-start ancestry differs')
        if cfg['repair_arm']=='warm' and cfg.get('repair_profile')=='dose_v1':
            from apt_v2.grpo_dose_study import recipe
            if read(initial/'source_config.json')!=recipe(cfg['warm_target_repeats']):
                raise ValueError('dose B source recipe differs')
        if cfg['repair_arm']=='warm' and cfg.get('repair_profile')=='lr_v1':
            from apt_v2.grpo_lr_study import recipe
            if read(initial/'source_config.json') != recipe(cfg['warm_learning_rate']):
                raise ValueError('learning-rate B source recipe differs')
    identity = read(initial / 'model_identity.json')
    if identity['resolved_revision'] != cfg['revision'] or identity['model_id'] != cfg['base_model']:
        raise ValueError('base revision differs')
    for name, sha in tqdm(identity['files_sha256'].items(), desc='Prepare base identity', file=sys.stdout, disable=False):
        if file_hash(under(ROOT, cfg['base_dir']) / name) != sha:
            raise ValueError('base model file differs: '+name)
    dest = output / 'initial'
    if dest.exists():
        raise FileExistsError('prepare incomplete; start a fresh run ID')
    shutil.copytree(initial, dest)
    tok = output / 'tokenizer'
    tok.mkdir()
    for name in identity['files_sha256']:
        if not name.endswith('.safetensors'):
            shutil.copyfile(under(ROOT, cfg['base_dir']) / name, tok / name)
    extra = []
    if cfg['phase'] == 'formal':
        from dataclasses import asdict
        from apt_v2.grpo_evaluation import development_tasks, evaluation_plan
        from apt_v2.dpo_recovery import build_context, SCENARIOS
        verify_files(ROOT, cfg['profile_binding'])
        profile = ROOT/cfg['profile_input_dir']
        if not read(profile/'audit.json')['passed'] or read(profile/'run_status.json')['error'] is not None:
            raise ValueError('audited online GPU profile required')
        shutil.copytree(profile, output/'profile')
        atomic_json(output/'development_tasks.json', [asdict(t) for t in development_tasks(cfg)])
        atomic_json(output/'recovery_contexts.json', [build_context(t, s, allowed_split='development')
                    for t in development_tasks(cfg) for s in SCENARIOS])
        atomic_json(output/'evaluation_plan.json', evaluation_plan(cfg))
        extra = ['development_tasks.json', 'recovery_contexts.json', 'evaluation_plan.json'] + [
            p.relative_to(output).as_posix() for p in (output/'profile').rglob('*') if p.is_file()]
    if cfg.get('curriculum_schema'):
        from apt_v2.grpo_curriculum import frozen_inputs
        for name,value in frozen_inputs(cfg).items():
            atomic_json(output/name,value)
            extra.append(name)
    if cfg.get('closing_evaluation',{}).get('enabled'):
        from apt_v2.period_closing import closing_tasks,expert_dependency_proof
        from dataclasses import asdict
        atomic_json(output/'closing_tasks.json',[asdict(t) for t in closing_tasks()])
        proof=[expert_dependency_proof(t) for t in closing_tasks()]
        if not all(p['passed'] for p in proof):raise ValueError('SQLite closing dependency proof failed')
        atomic_json(output/'closing_dependency_proof.json',proof)
        extra += ['closing_tasks.json','closing_dependency_proof.json']
    names = extra + ['config.json', 'execution_environment.json'] + [p.relative_to(output).as_posix()
             for folder in (dest, tok) for p in folder.rglob('*') if p.is_file()]
    atomic_json(output / 'input_binding.json', {'files': {n: file_hash(output/n) for n in names},
                'config_sha256': file_hash(output/'config.json'), 'initial_adapter_sha256': cfg['adapter_sha256'],
                'source_sft_updates': cfg['source_sft_updates'], 'source_dpo_updates': cfg['source_dpo_updates'], 'initial_rl_updates': 0})
    verify_inputs(output)
