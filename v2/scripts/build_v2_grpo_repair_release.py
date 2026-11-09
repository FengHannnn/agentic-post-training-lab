"""Build and hash-verify the six-hour gated SQL-repair study cloud archive."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,file_hash,verify_files,validate_config
from run_v2_grpo import source_files
from run_v2_grpo_repair import MANIFEST


def build(output_dir):
    cfg=read(ROOT/'configs/grpo-sql-repair-study-v1.json');branch=read(ROOT/cfg['branch_template']);validate_config(branch)
    initial=ROOT/branch['initial_dir'];verify_files(initial/'adapter',read(initial/'complete.json')['adapter_sha256'])
    if file_hash(initial/'adapter/adapter_model.safetensors')!=branch['baseline_W_sha256']:raise ValueError('W source identity differs')
    validation=read(ROOT/cfg['validation_report'])
    if not validation['passed'] or not validation.get('full_regression',{}).get('passed'):
        raise ValueError('CPU validation and full regression must pass before release')
    files=source_files()
    for folder in (initial,ROOT/branch['profile_input_dir'],ROOT/'tests/fixtures',(ROOT/cfg['sft_examples']).parent):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pt':files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for name in (cfg['contract_report'],cfg['validation_report'],'README.md','PROJECT_REQUIREMENTS.md',
                 'reports/grpo-sql-repair-preparation-v1-20261109.md'):files[name]=file_hash(ROOT/name)
    manifest={'run_id':cfg['run_id'],'study_phase':'repair_study','files':files,'initial_W_sha256':branch['baseline_W_sha256'],
              'maximum_job_wall_seconds':21600,'diagnostic_episodes':96,'maximum_groups_per_branch':36,
              'verifier_version':'sql_semantics_v2','local_validation':validation,
              'standard_episodes_per_model':132,'compound_episodes_per_model':16,'quality_actions_per_model':14}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True);archive=folder/(cfg['run_id']+'-run.zip')
    manifest_text=json.dumps(manifest,ensure_ascii=False,indent=2)
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for rel in tqdm(sorted(files),desc='Build cloud ZIP',unit='file',file=sys.stdout,disable=False):z.write(ROOT/rel,'v2/'+rel)
        z.writestr('v2/'+MANIFEST,manifest_text)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();expected={'v2/'+n for n in files}|{'v2/'+MANIFEST}
        if z.testzip() is not None or set(names)!=expected or len(names)!=len(expected):raise ValueError('archive CRC/unique entry coverage differs')
        for rel,sha in tqdm(files.items(),desc='Verify archive hashes',unit='file',file=sys.stdout,disable=False):
            if hashlib.sha256(z.read('v2/'+rel)).hexdigest()!=sha:raise ValueError('archived hash differs: '+rel)
    verify_files(ROOT,files)
    (ROOT/MANIFEST).write_bytes(manifest_text.encode('utf8'))
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'sha256':file_hash(archive),'bytes':archive.stat().st_size,
             'entries':len(files)+1,'bound_files':len(files),'crc_and_hashes_passed':True,'initial_W_sha256':branch['baseline_W_sha256']}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.output_dir)
