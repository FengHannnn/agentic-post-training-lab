"""Build the frozen matched fine-credit experiment and verify every ZIP member."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from grpo_common import ROOT,read,file_hash,verify_files
from grpo_fine_common import matched_configs
from run_v2_grpo import source_files


def build(output_dir):
    cfg,branches=matched_configs(ROOT/'configs/grpo-fine-credit-v1.json');base=branches['legacy']
    initial=ROOT/base['initial_dir'];complete=read(initial/'complete.json');verify_files(initial/'adapter',complete['adapter_sha256'])
    if file_hash(initial/'adapter/adapter_model.safetensors')!=base['adapter_sha256']:raise ValueError('frozen W differs')
    files=source_files()
    for folder in (initial,ROOT/base['profile_input_dir'],ROOT/'tests/fixtures'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for p in sorted((ROOT/'hotfixes').rglob('*.py')):
        files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for name in [cfg['contract_report'],cfg['validation_report']]:
        files[name]=file_hash(ROOT/name)
    verify_files(ROOT,base['profile_binding'])
    validation=read(ROOT/cfg['validation_report'])
    if not validation['passed']:raise ValueError('local validation must pass')
    manifest={'run_id':cfg['run_id'],'study_phase':'fine_credit_ablation','files':files,'initial_adapter_sha256':base['adapter_sha256'],
              'maximum_job_wall_seconds':cfg['max_wall_seconds'],'validation':validation,
              'clone_policy':cfg['clone_policy'],'groups_per_branch_maximum':48,
              'budget_policy':cfg['budget_policy'],'common_update_probe':12,
              'extra_sft_updates':0,'extra_dpo_updates':0,'methods':{b:{'credit_method':c['credit_method'],'learning_rate':c['learning_rate']} for b,c in branches.items()}}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True);archive=folder/(cfg['run_id']+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(files):z.write(ROOT/name,'v2/'+name)
        z.writestr('v2/grpo_fine_release_manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        expected={'v2/'+n for n in files}|{'v2/grpo_fine_release_manifest.json'}
        if z.testzip() is not None or set(z.namelist())!=expected or len(z.namelist())!=len(expected):raise ValueError('fine credit archive integrity/coverage differs')
        for name,sha in files.items():
            if hashlib.sha256(z.read('v2/'+name)).hexdigest()!=sha:raise ValueError('fine credit archived binding differs')
    verify_files(ROOT,files)
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'sha256':file_hash(archive),'bytes':archive.stat().st_size,'entries':len(files)+1,'bound_files':len(files),'crc_and_hashes_passed':True,'initial_W_sha256':base['adapter_sha256']}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();build(a.output_dir)
