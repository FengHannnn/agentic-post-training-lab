"""Bounded C2 GPU preflight with terminal progress, exact transactions and failure ZIPs."""
import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from grpo_common import (ROOT, read, atomic_json, file_hash, verify_files, validate_config,
                         initial_policy, load_transition)
from streamed_process import run_logged


def resource_snapshot(output, name):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,memory.used,memory.free,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        value = {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as exc:
        value = {'error': str(exc)}
    atomic_json(output/'resources'/f'{name}-{time.time_ns()}.json', value)


def source_files():
    roots = [('src', '*.py'), ('scripts', '*.py'), ('configs', '*.json'), ('tests', '*.py')]
    paths = [p for directory, pattern in roots for p in sorted((ROOT/directory).rglob(pattern))]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def package_review(output, archive):
    excluded = ('.pt', '.pyc')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p.suffix not in excluded and '.writing' not in p.name:
                z.write(p, p.relative_to(output.parent).as_posix())


def run(config=None, resume=False):
    from tqdm.auto import tqdm
    config = config or ROOT/'configs/grpo-outcome-preflight-v2.json'
    cfg = read(config)
    validate_config(cfg)
    output = ROOT/'artifacts'/cfg['run_id']
    release = read(ROOT/'grpo_release_manifest.json')
    verify_files(ROOT, release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:
        raise ValueError('config must be in the published release')
    if resume:
        if not (output/'input_binding.json').exists() or read(output/'config.json') != cfg:
            raise ValueError('resume requires an identically configured prepared run')
        verify_files(ROOT, read(output/'run_manifest.json')['source_sha256'])
        prior = read(output/'runner_state.json')
        spent = max(prior.get('elapsed_seconds', 0), read(output/'run_status.json').get('elapsed_seconds', 0)
                    if (output/'run_status.json').exists() else 0)
        attempt = prior.get('attempt', 1)+1
    else:
        if output.exists() or (ROOT/(cfg['run_id']+'-review.zip')).exists():
            raise FileExistsError('run already exists; use --resume for its committed checkpoint')
        output.mkdir(parents=True)
        atomic_json(output/'config.json', cfg)
        sources = source_files()
        atomic_json(output/'run_manifest.json', {'source_sha256': sources})
        for name in sources:
            p = output/'source'/name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, p)
        spent, attempt = 0., 1
    started = time.monotonic()
    stages, error, stop_reason = [], None, None
    state = initial_policy(cfg)
    cursor = 0
    committed_states = {0: dict(state)}
    # Recover any transaction committed just before a driver interruption.
    while (output/'transitions'/f'group-{cursor:04}.json').exists():
        state = load_transition(output, cursor, state)['after']
        cursor += 1
        committed_states[cursor] = dict(state)
    runner = {'policy': state, 'next_group': cursor, 'elapsed_seconds': spent,
              'attempt': attempt, 'refresh_only_last_group': False}
    atomic_json(output/'runner_state.json', runner)
    archive = ROOT/(cfg['run_id']+('-review.zip' if attempt == 1 else f'-review-attempt-{attempt:03}.zip'))
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_NO_USAGE_STATS='1', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    extra_evaluations = 1+len(cfg['group_evaluation_cursors']) if cfg['phase'] == 'formal' else 0
    progress = tqdm(total=2*cfg['max_groups']+3+extra_evaluations, desc='C2 '+cfg['phase'], unit='stage', file=sys.stdout,
                    disable=False, dynamic_ncols=True)
    def save_runner():
        runner.update(policy=state, next_group=cursor, elapsed_seconds=spent+time.monotonic()-started)
        atomic_json(output/'runner_state.json', runner)
    def stage(name, script, args):
        save_runner()
        remaining = cfg['max_wall_seconds']-cfg['packaging_reserve_seconds']-(spent+time.monotonic()-started)
        if remaining <= 0:
            raise TimeoutError('C2 job wall-time budget reached')
        log = output/f'{name}-attempt-{attempt:03}.log'
        print(f'\nStarting {name}; remaining {remaining:.0f}s; log: {log}', flush=True)
        resource_snapshot(output, name+'-before')
        stage_started = time.monotonic()
        code = run_logged([sys.executable, '-u', str(ROOT/'scripts'/script), *args, '--output', str(output)],
                          cwd=ROOT, env=env, log_path=log, timeout=remaining)
        stages.append({'stage': name, 'exit_code': code, 'elapsed_seconds': time.monotonic()-stage_started})
        save_runner()
        resource_snapshot(output, name+'-after')
        if code:
            raise RuntimeError(f'{name} failed; inspect {log.name}')
        progress.set_postfix(stage=name, updates=state['updates'], refresh=False)
        progress.update(1)
    checked_evaluations = set()
    def evaluate_checkpoint(label, evaluation_cursor, policy):
        if label in checked_evaluations:
            return
        spec = {'label': label, 'group_cursor': evaluation_cursor, 'policy': policy}
        path = output/'evaluations'/(label+'-policy.json')
        if path.exists():
            if read(path) != spec:
                raise ValueError('frozen evaluation policy spec differs')
        else:
            atomic_json(path, spec)
        stage('evaluate-'+label, 'evaluate_v2_grpo.py', ['--label', label, '--spec', str(path)])
        checked_evaluations.add(label)
    def evaluate_due():
        if cfg['phase'] != 'formal':
            return
        for n in cfg['group_evaluation_cursors']:
            label = f'grpo_g{n:04}'
            if n <= cursor and label not in checked_evaluations:
                evaluate_checkpoint(label, n, committed_states[n])
    try:
        if not (output/'input_binding.json').exists():
            stage('prepare', 'grpo_worker.py', ['prepare'])
        if cfg['phase'] == 'formal':
            evaluate_checkpoint('weighted_sft', 0, committed_states[0])
            evaluate_due()
        for index in range(cursor, cfg['max_groups']):
            if state['updates'] >= cfg['max_updates']:
                stop_reason = 'update_cap'
                break
            policy_file = output/'policies'/f'before-group-{index:04}.json'
            atomic_json(policy_file, state)
            args = ['--group-index', str(index), '--policy-file', str(policy_file)]
            stage(f'collect-{index:04}', 'grpo_worker.py', ['collect', *args])
            stage(f'update-{index:04}', 'grpo_worker.py', ['update', *args])
            result = load_transition(output, index, state)
            state = result['after']
            cursor = index+1
            committed_states[cursor] = dict(state)
            save_runner()
            evaluate_due()
        else:
            stop_reason = 'group_cap'
        if state['updates']:
            # Sample the final saved version even when the last training group reached the cap.
            policy_file = output/'policies'/f'before-group-{cursor:04}.json'
            atomic_json(policy_file, state)
            runner['refresh_only_last_group'] = True
            save_runner()
            stage('refresh-sample', 'grpo_worker.py', ['collect', '--group-index', str(cursor),
                                                     '--policy-file', str(policy_file)])
        save_runner()
        stage('audit', 'audit_v2_grpo_formal.py' if cfg['phase'] == 'formal' else 'audit_v2_grpo.py', [])
        verify_files(ROOT, read(output/'run_manifest.json')['source_sha256'])
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        stop_reason = 'timeout' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired)) else 'failed'
    finally:
        save_runner()
        if error is None:
            progress.total = progress.n
            progress.refresh()
        progress.close()
        status = {'run_id': cfg['run_id'], 'phase': cfg['phase'], 'error': error, 'stop_reason': stop_reason,
                  'elapsed_seconds': spent+time.monotonic()-started, 'stages': stages,
                  'actual_updates': state['updates'], 'committed_groups': cursor, 'attempt': attempt,
                  'max_wall_seconds': cfg['max_wall_seconds'], 'maximum_updates': cfg['max_updates'],
                  'state_pt_kept_on_cloud': True, 'archive': archive.name,
                  'formal_job_index': cfg.get('formal_job_index'),
                  'evaluation_cursors': cfg.get('group_evaluation_cursors', [])}
        atomic_json(output/'run_status.json', status)
        atomic_json(output/'attempts'/f'attempt-{attempt:03}-status.json', status)
        package_review(output, archive)
        print('\nDownload: '+str(archive), flush=True)
        print('Run status: '+json_status(status), flush=True)
    if error:
        raise RuntimeError(error)
    return archive


def json_status(status):
    import json
    return json.dumps({k: status[k] for k in ('error', 'stop_reason', 'actual_updates', 'committed_groups')})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    run(args.config, args.resume)
