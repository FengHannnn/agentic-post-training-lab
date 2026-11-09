"""Install the audited quality-outcome recording repair without resetting a GRPO run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

FIX_ID = 'alias-quality-outcome-fix-v1'
RUN_ID = 'grpo-alias-curriculum-v1-20261109'
TARGETS = ('scripts/evaluate_v2_grpo.py', 'scripts/audit_v2_grpo_alias.py', 'scripts/run_v2_grpo_alias_curriculum.py', 'tests/test_grpo_alias_quality_v2.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.quality-fix-writing')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def safe(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('path outside project root')
    return path


def protected_files(output):
    result = {}
    for path in sorted(output.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(output)
        if 'source' in relative.parts or relative.parts[0] == 'hotfixes' or path.name == 'run_manifest.json':
            continue
        result[relative.as_posix()] = sha(path)
    return result


def apply(root, bundle=None):
    root = Path(root).resolve()
    bundle = Path(bundle or Path(__file__).resolve().parent)
    patch = read(bundle / 'patch.json')
    if patch['fix_id'] != FIX_ID or tuple(patch['targets']) != TARGETS:
        raise ValueError('unexpected quality-outcome patch scope')
    for relative, expected in patch['bundle_files'].items():
        if sha(safe(root, relative)) != expected:
            raise ValueError('uploaded patch file differs: ' + relative)
    output = root / 'artifacts' / RUN_ID
    release_path = root / 'grpo_alias_release_manifest.json'
    locations = [output, output / 'branches/control', output / 'branches/curriculum']
    record_dir = output / 'hotfixes' / FIX_ID
    receipt_path = record_dir / 'receipt.json'
    if receipt_path.exists():
        for relative in TARGETS:
            expected = patch['targets'][relative]['after']
            if sha(root / relative) != expected:
                raise ValueError('previously repaired runtime differs')
            for location in locations:
                if (sha(location / 'source' / relative) != expected
                        or read(location / 'run_manifest.json')['source_sha256'][relative] != expected):
                    raise ValueError('previously repaired frozen source differs')
        print(json.dumps({'already_applied': True, 'fix_id': FIX_ID}))
        return read(receipt_path)

    # Validate the original published release and every frozen source before writing.
    if sha(release_path) != patch['original_release_manifest_sha256']:
        raise ValueError('original alias release manifest differs')
    release = read(release_path)
    for relative, expected in release['files'].items():
        if sha(safe(root, relative)) != expected:
            raise ValueError('original release file differs: ' + relative)
    status = read(output / 'run_status.json')
    if (status['committed_groups'] != {'control': 48, 'curriculum': 48}
            or status['attempt'] != 2 or 'evaluate-control-grpo_g0048' not in status['error']):
        raise ValueError('expected committed-control-evaluation failure required')
    if sha(output/'branches/control/evaluations/grpo_g0048.json')!=patch['expected_control_evaluation_sha256']:
        raise ValueError('committed control evaluation differs')
    if read(output / 'config.json') != read(root / 'configs/grpo-alias-curriculum-v1.json'):
        raise ValueError('run configuration differs')
    planned = {}
    for relative in TARGETS:
        change = patch['targets'][relative]
        before=sha(root/relative) if (root/relative).exists() else None
        if before != change['before'] or sha(bundle / 'payload' / relative) != change['after']:
            raise ValueError('runtime or payload differs: ' + relative)
        planned[root / relative] = (bundle / 'payload' / relative).read_bytes()
    migration = {'fix_id': FIX_ID, 'reason': 'preserve failed quality outcomes, finish the paired evaluation, retain original acceptance gates',
                 'targets': patch['targets'], 'training_algorithm_changed': False}
    for location in locations:
        manifest_path = location / 'run_manifest.json'
        manifest = read(manifest_path)
        for relative, expected in manifest['source_sha256'].items():
            if sha(root / relative) != expected or sha(location / 'source' / relative) != expected:
                raise ValueError('original runtime/frozen source differs: ' + relative)
        for relative in TARGETS:
            if manifest['source_sha256'].get(relative) != patch['targets'][relative]['before']:
                raise ValueError('original target binding differs')
            if patch['targets'][relative]['before'] is None and (location/'source'/relative).exists():
                raise ValueError('unexpected new frozen target already exists')
            planned[location / 'source' / relative] = (bundle / 'payload' / relative).read_bytes()
            manifest['source_sha256'][relative] = patch['targets'][relative]['after']
        manifest.setdefault('source_migrations', []).append(migration)
        planned[manifest_path] = encoded(manifest)
    for relative in TARGETS:
        release['files'][relative] = patch['targets'][relative]['after']
    release['files'].update(patch['bundle_files'])
    release['files'][(bundle / 'patch.json').resolve().relative_to(root).as_posix()] = sha(bundle / 'patch.json')
    release.setdefault('source_migrations', []).append(migration)
    planned[release_path] = encoded(release)
    preserved = protected_files(output)
    backups = {}
    # Keep original published/frozen files alongside the run's repair record.
    for path in planned:
        relative = path.relative_to(root)
        # Short names also keep Windows verification fixtures within path limits.
        key = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16] + '.bak'
        backup = record_dir / 'before' / key
        if not path.exists():
            backups[relative.as_posix()]={'sha256':None,'backup_file':None}
            continue
        if backup.exists():
            if sha(backup) != sha(path):
                raise ValueError('partial repair backup differs; inspect before retry')
        else:
            write(backup, path.read_bytes())
        backups[relative.as_posix()] = {'sha256': sha(path), 'backup_file': backup.relative_to(output).as_posix()}
    previous={path:path.read_bytes() if path.exists() else None for path in planned}
    try:
        for path, data in planned.items():
            write(path, data)
    except BaseException:
        for path,data in previous.items():
            if data is not None:write(path,data)
            elif path.exists():path.unlink()
        raise
    if protected_files(output) != preserved:
        raise ValueError('training/evaluation artifacts changed during repair')
    for location in locations:
        for relative, expected in read(location / 'run_manifest.json')['source_sha256'].items():
            if sha(root / relative) != expected or sha(location / 'source' / relative) != expected:
                raise ValueError('repaired source binding verification failed')
    for relative, expected in read(release_path)['files'].items():
        if sha(root / relative) != expected:
            raise ValueError('repaired release verification failed')
    write(record_dir / 'protected_files.json', encoded(preserved))
    receipt = {'passed': True, 'fix_id': FIX_ID, 'original_files_backed_up': backups,
               'protected_files_verified_unchanged': len(preserved), 'actual_updates': status['actual_updates'],
               'committed_groups': status['committed_groups'], 'elapsed_seconds_preserved': status['elapsed_seconds'],
               'remaining_budget_seconds': status['max_wall_seconds'] - status['elapsed_seconds'],
               'training_restarted': False, 'config_changed': False, 'source_migration': migration}
    write(receipt_path, encoded(receipt))
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    apply(args.root)
