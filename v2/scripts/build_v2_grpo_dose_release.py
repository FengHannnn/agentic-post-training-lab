"""Build and verify the published warm-start search release."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,file_hash,verify_files
from run_v2_grpo import source_files
from grpo_dose_worker import protocol_for
from apt_v2 import grpo_dose_study, grpo_lr_study, grpo_lr_midpoint, grpo_closeout_study


def build(output_dir, config=None):
    config = config or ROOT/'configs/grpo-sql-dose-study-v1.json'
    cfg = read(config)
    protocol = protocol_for(cfg)
    protocol.validate_master(cfg)
    protocol.validate_prior(read(ROOT/cfg['input_dir']/'prior-evidence.json'), cfg)
    manifest_name = protocol.MANIFEST
    validation=read(ROOT/cfg['validation_report'])
    if not validation['passed'] or not validation.get('regression',{}).get('passed'):raise ValueError('warm-start validation and regression must pass')
    files=source_files();branch=read(ROOT/cfg['branch_template'])
    if config.resolve().relative_to(ROOT).as_posix() not in files:
        raise ValueError('master config must be included in the release source')
    for folder in (ROOT/cfg['input_dir'],ROOT/'tests/fixtures',ROOT/branch['initial_dir'],ROOT/branch['profile_input_dir'],
                   ROOT/'artifacts/inputs/grpo-sql-repair-20261109', ROOT/'artifacts/inputs/grpo-sql-retention-20261110',
                   ROOT/'artifacts/inputs/grpo-warmup-probe-20261110', ROOT/grpo_dose_study.INPUT_DIR,
                   ROOT/grpo_lr_study.INPUT_DIR, ROOT/grpo_lr_midpoint.INPUT_DIR, ROOT/grpo_closeout_study.INPUT_DIR):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pyc') and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    reports = [cfg['contract_report'], cfg['validation_report'], 'README.md', 'PROJECT_REQUIREMENTS.md']
    if protocol is grpo_dose_study:
        reports.extend(['reports/grpo-sql-dose-preparation-v1-20261110.md',
                        'reports/grpo-warmup-probe-results-v1-20261110.md',
                        'reports/grpo-warmup-probe-results-validation-v1-20261110.json'])
        experiment = {'target_repeats': cfg['target_repeats']}
    else:
        reports.extend([cfg['preparation_report'], 'reports/grpo-sql-dose-results-v1-20261111.md'])
        experiment = {'target_repeats': cfg['target_repeats'], 'learning_rates': cfg['learning_rates'],
                      'max_wall_seconds': cfg['max_wall_seconds']}
    if protocol is grpo_lr_midpoint:
        reports.append('reports/grpo-sql-lr-results-v1-20261111.md')
    if protocol is grpo_closeout_study:
        reports.append('reports/grpo-sql-lr-midpoint-results-v1-20261112.md')
    for name in reports:
        files[name] = file_hash(ROOT/name)
    manifest = {'run_id': cfg['run_id'], 'files': files, 'new_authorized_seconds': cfg['new_authorized_seconds'],
                'prior_elapsed_seconds': cfg['prior_elapsed_seconds'], 'total_authorized_seconds': cfg['total_authorized_seconds'],
                **experiment, 'run_grpo': False, 'local_validation': validation}
    content=json.dumps(manifest,ensure_ascii=False,indent=2);folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    archive=folder/(cfg['run_id']+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n in tqdm(sorted(files),desc='Build warm-start ZIP',unit='file',file=sys.stdout,disable=False):z.write(ROOT/n,'v2/'+n)
        z.writestr('v2/'+manifest_name,content)
    with zipfile.ZipFile(archive) as z:
        expected={'v2/'+n for n in files}|{'v2/'+manifest_name}
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=expected:raise ValueError('dose ZIP CRC/coverage differs')
        for n,sha in tqdm(files.items(),desc='Verify release hashes',unit='file',file=sys.stdout,disable=False):
            if hashlib.sha256(z.read('v2/'+n)).hexdigest()!=sha:raise ValueError('archived dose file differs: '+n)
    verify_files(ROOT,files);(ROOT/manifest_name).write_bytes(content.encode('utf8'))
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),
             'entries':len(files)+1,'bound_files':len(files),'crc_and_hashes_passed':True,'new_authorized_seconds':cfg['new_authorized_seconds'],'max_wall_seconds':cfg['max_wall_seconds']}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return archive


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config',type=Path)
    a=p.parse_args();build(a.output_dir,a.config)
