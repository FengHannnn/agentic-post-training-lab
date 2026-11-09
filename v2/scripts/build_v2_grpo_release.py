"""Create the frozen C2 preflight ZIP including reviewed W, without base weights."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path
from grpo_common import ROOT, read, atomic_json, file_hash, verify_files, validate_config
from run_v2_grpo import source_files


def build(output_dir, source_w, config=None):
    cfg = read(config or ROOT/'configs/grpo-outcome-preflight-v2.json')
    validate_config(cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_w = Path(source_w).resolve()
    complete = read(source_w/'final/complete.json')
    verify_files(source_w/'final/adapter', complete['adapter_sha256'])
    if file_hash(source_w/'final/adapter/adapter_model.safetensors') != cfg['adapter_sha256']:
        raise ValueError('reviewed W input differs')
    if not read(source_w/'audit.json')['passed']:
        raise ValueError('audited weighted SFT required')
    initial = ROOT/cfg['initial_dir']
    mapping = {**{'adapter/'+n: source_w/'final/adapter'/n for n in complete['adapter_sha256']},
               'complete.json': source_w/'final/complete.json', 'model_identity.json': source_w/'model_identity.json',
               'source_config.json': source_w/'config.json', 'source_audit.json': source_w/'audit.json'}
    for name, source in mapping.items():
        target = initial/name
        if target.exists():
            if file_hash(target) != file_hash(source):
                raise FileExistsError('existing frozen W input differs: '+str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    files = source_files()
    for name, sha in cfg.get('profile_binding', {}).items():
        if file_hash(ROOT/name) != sha:
            raise ValueError('reviewed GPU profile binding differs')
        files[name] = sha
    for p in sorted(initial.rglob('*')):
        if p.is_file():
            files[p.relative_to(ROOT).as_posix()] = file_hash(p)
    for name in (cfg.get('contract_report', 'reports/grpo-c2-contract-v1-20261105.md'),
                 cfg.get('validation_report', 'reports/grpo-c2-local-validation-v1-20261105.json')):
        files[name] = file_hash(ROOT/name)
    manifest = {'run_id': cfg['run_id'], 'files': files, 'algorithm': cfg['algorithm'],
                'initial_adapter_sha256': cfg['adapter_sha256'], 'model_revision': cfg['revision'],
                'source_sft_updates': 513, 'source_dpo_updates': 11, 'maximum_gpu_wall_seconds': cfg['max_wall_seconds'],
                'validation': read(ROOT/cfg.get('validation_report', 'reports/grpo-c2-local-validation-v1-20261105.json'))}
    atomic_json(ROOT/'grpo_release_manifest.json', manifest)
    archive = output_dir/(cfg['run_id']+'-run.zip')
    if archive.exists():
        raise FileExistsError('release archive already exists: '+str(archive))
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):
            z.write(ROOT/name, 'v2/'+name)
        z.write(ROOT/'grpo_release_manifest.json', 'v2/grpo_release_manifest.json')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or any('..' in Path(n).parts or n.startswith(('/', '\\')) for n in z.namelist()):
            raise ValueError('release archive integrity/path check failed')
        expected = set('v2/'+n for n in files) | {'v2/grpo_release_manifest.json'}
        if len(z.namelist()) != len(expected) or set(z.namelist()) != expected:
            raise ValueError('release archive coverage differs')
        import hashlib
        for name, sha in files.items():
            if hashlib.sha256(z.read('v2/'+name)).hexdigest() != sha:
                raise ValueError('archived source binding differs: '+name)
    verify_files(ROOT, files)
    receipt = {'archive': archive.name, 'sha256': file_hash(archive), 'bytes': archive.stat().st_size,
               'archive_entries': len(files)+1, 'bound_files': len(files), 'crc_passed': True,
               'archived_hashes_passed': True, 'path_check_passed': True,
               'initial_adapter_sha256': cfg['adapter_sha256'], 'run_id': cfg['run_id']}
    atomic_json(output_dir/(cfg['run_id']+'-release.json'), receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--source-w', type=Path, required=True)
    p.add_argument('--config', type=Path)
    args = p.parse_args()
    build(args.output_dir, args.source_w, args.config)
