"""CRC/hash-verified release for the pre-registered target-dose warm-up search."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,file_hash,verify_files
from run_v2_grpo import source_files
from run_v2_grpo_dose import MANIFEST
from apt_v2.grpo_dose_study import validate_master,validate_prior


def build(output_dir):
    cfg=read(ROOT/'configs/grpo-sql-dose-study-v1.json');validate_master(cfg);validate_prior(read(ROOT/cfg['input_dir']/'prior-evidence.json'),cfg)
    validation=read(ROOT/cfg['validation_report'])
    if not validation['passed'] or not validation.get('regression',{}).get('passed'):raise ValueError('dose validation and regression must pass')
    files=source_files();branch=read(ROOT/cfg['branch_template'])
    for folder in (ROOT/cfg['input_dir'],ROOT/'tests/fixtures',ROOT/branch['initial_dir'],ROOT/branch['profile_input_dir'],
                   ROOT/'artifacts/inputs/grpo-sql-repair-20261109',ROOT/'artifacts/inputs/grpo-sql-retention-20261110',ROOT/'artifacts/inputs/grpo-warmup-probe-20261110'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pyc') and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for n in (cfg['contract_report'],cfg['validation_report'],'README.md','PROJECT_REQUIREMENTS.md',
              'reports/grpo-sql-dose-preparation-v1-20261110.md','reports/grpo-warmup-probe-results-v1-20261110.md',
              'reports/grpo-warmup-probe-results-validation-v1-20261110.json'):files[n]=file_hash(ROOT/n)
    manifest={'run_id':cfg['run_id'],'files':files,'new_authorized_seconds':21600,'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],
              'total_authorized_seconds':cfg['total_authorized_seconds'],'target_repeats':[4,8,16],'run_grpo':False,'local_validation':validation}
    content=json.dumps(manifest,ensure_ascii=False,indent=2);folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    archive=folder/(cfg['run_id']+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for n in tqdm(sorted(files),desc='Build target-dose ZIP',unit='file',file=sys.stdout,disable=False):z.write(ROOT/n,'v2/'+n)
        z.writestr('v2/'+MANIFEST,content)
    with zipfile.ZipFile(archive) as z:
        expected={'v2/'+n for n in files}|{'v2/'+MANIFEST}
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=expected:raise ValueError('dose ZIP CRC/coverage differs')
        for n,sha in tqdm(files.items(),desc='Verify dose ZIP hashes',unit='file',file=sys.stdout,disable=False):
            if hashlib.sha256(z.read('v2/'+n)).hexdigest()!=sha:raise ValueError('archived dose file differs: '+n)
    verify_files(ROOT,files);(ROOT/MANIFEST).write_bytes(content.encode('utf8'))
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),
             'entries':len(files)+1,'bound_files':len(files),'crc_and_hashes_passed':True,'new_authorized_seconds':21600}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.output_dir)
