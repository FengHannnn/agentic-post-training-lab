"""Build and CRC/hash-verify the short warm-up diagnosis cloud archive."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,file_hash,verify_files
from run_v2_grpo import source_files
from run_v2_grpo_warmup_probe import MANIFEST
from apt_v2.grpo_warmup_probe import validate_config,validate_prior


def build(output_dir):
    cfg=read(ROOT/'configs/grpo-sql-warmup-probe-v1.json');validate_config(cfg)
    validate_prior(read(ROOT/cfg['input_dir']/'prior-evidence.json'),cfg)
    validation=read(ROOT/cfg['validation_report'])
    if not validation['passed'] or not validation.get('regression',{}).get('passed'):raise ValueError('probe validation/tests must pass')
    files=source_files();branch=read(ROOT/cfg['branch_template'])
    for folder in (ROOT/cfg['input_dir'],ROOT/'tests/fixtures',ROOT/'artifacts/inputs/grpo-w-20261105',
                   ROOT/'artifacts/inputs/grpo-sql-repair-20261109',ROOT/'artifacts/inputs/grpo-sql-retention-20261110',ROOT/branch['profile_input_dir']):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pyc') and '__pycache__' not in p.parts:
                files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for name in (cfg['contract_report'],cfg['validation_report'],'README.md','PROJECT_REQUIREMENTS.md',
                 'reports/grpo-warmup-probe-preparation-v1-20261110.md',
                 'reports/grpo-sql-retention-results-v1-20261110.md','reports/grpo-sql-retention-results-validation-v1-20261110.json',
                 'reports/grpo-sql-repair-results-v1-20261110.md','reports/grpo-sql-repair-results-validation-v1-20261110.json'):
        files[name]=file_hash(ROOT/name)
    manifest={'run_id':cfg['run_id'],'files':files,'maximum_job_wall_seconds':1800,
              'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],'total_authorized_seconds':21600,
              'models':cfg['models'],'local_validation':validation,'optimizer_updates':0}
    content=json.dumps(manifest,ensure_ascii=False,indent=2);folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True)
    archive=folder/(cfg['run_id']+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for rel in tqdm(sorted(files),desc='Build warm-up probe ZIP',unit='file',file=sys.stdout,disable=False):z.write(ROOT/rel,'v2/'+rel)
        z.writestr('v2/'+MANIFEST,content)
    with zipfile.ZipFile(archive) as z:
        expected={'v2/'+p for p in files}|{'v2/'+MANIFEST}
        if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=expected:
            raise ValueError('probe archive CRC or unique coverage differs')
        for rel,sha in tqdm(files.items(),desc='Verify probe ZIP hashes',unit='file',file=sys.stdout,disable=False):
            if hashlib.sha256(z.read('v2/'+rel)).hexdigest()!=sha:raise ValueError('archived file hash differs: '+rel)
    verify_files(ROOT,files);(ROOT/MANIFEST).write_bytes(content.encode('utf8'))
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),
             'entries':len(files)+1,'bound_files':len(files),'crc_and_hashes_passed':True,'optimizer_updates':0}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.output_dir)
