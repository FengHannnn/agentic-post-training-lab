"""Publish C4 frozen sources and the three preselected immutable adapters."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from grpo_c4_common import ROOT,read,file_hash,verify_files,validate_config,LABELS,ADAPTERS,VERSIONS,confirmation_plan
from run_v2_grpo import source_files

C3_REVIEW_SHA='c6aa4683dab934e5ecf2dd596d42747884c05b9cdb04dc7beb23dc9265ca972e'


def build(output_dir,c3_review_zip,config=None):
    cfg=validate_config(read(config or ROOT/'configs/grpo-c4-confirmation-v1.json'));validation=read(ROOT/cfg['validation_report'])
    if not validation['passed']:raise ValueError('C4 local validation must pass before publication')
    archive_source=Path(c3_review_zip)
    if file_hash(archive_source)!=C3_REVIEW_SHA:raise ValueError('audited C3 review ZIP identity differs')
    payload={};provenance={}
    def add_candidate(label,files,source):
        sha=hashlib.sha256(files['adapter_model.safetensors']).hexdigest()
        if sha!=ADAPTERS[label]:raise ValueError('candidate adapter hash differs: '+label)
        adapter_cfg=json.loads(files['adapter_config.json'])
        if adapter_cfg['r']!=8 or adapter_cfg['lora_alpha']!=16 or set(adapter_cfg['target_modules'])!={'q_proj','v_proj'} or adapter_cfg['lora_dropout']!=0:raise ValueError('fixed LoRA structure differs')
        provenance[label]={'label':label,'adapter_sha256':sha,'version':VERSIONS[label],'ancestry':{'sft':513,'dpo':11,'rl':VERSIONS[label]},'preselected_before_C4':True,'source':source}
        folder=cfg['models'][label]['input_dir']
        for name,content in files.items():payload[folder+'/adapter/'+name]=content
        payload[folder+'/provenance.json']=json.dumps(provenance[label],ensure_ascii=False,indent=2).encode()
    for label,folder in [('weighted_sft','grpo-w-20261105'),('outcome_rl16','grpo-outcome-c2-rl16-20261106')]:
        source=ROOT/'artifacts/inputs'/folder
        if label=='outcome_rl16':
            selection=read(source/'selection.json')
            if not selection['audits']['local_passed'] or selection['policy']['adapter_sha256']!=ADAPTERS[label]:raise ValueError('audited C2 selection differs')
        add_candidate(label,{p.name:p.read_bytes() for p in sorted((source/'adapter').iterdir()) if p.is_file()}, {'project_input':source.relative_to(ROOT).as_posix(),'review_sha256':read(source/'selection.json')['review_zip_sha256'] if label=='outcome_rl16' else None})
    with zipfile.ZipFile(archive_source) as z:
        if z.testzip() is not None:raise ValueError('C3 review CRC failure')
        base='grpo-c3-matched-v1-20261106/';a=json.loads(z.read(base+'c3_audit.json'))
        if not a['passed'] or a['training']['fine']['actual_updates']!=17:raise ValueError('audited C3 fine selection differs')
        prefix=base+'branches/fine/checkpoints/group-0021/adapter/'
        files={n[len(prefix):]:z.read(n) for n in z.namelist() if n.startswith(prefix) and '/' not in n[len(prefix):] and not n.endswith('/')}
        add_candidate('fine_rl17',files,{'review_sha256':C3_REVIEW_SHA,'archive_checkpoint':prefix.rstrip('/')})
    files=source_files()
    for folder in [ROOT/'tests/fixtures']:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for name in [cfg['contract_report'],cfg['validation_report'],cfg['base_identity_file'],'hotfixes/grpo-native-eos-v1/apply.py']:
        files[name]=file_hash(ROOT/name)
    for name,sha in files.items():payload[name]=(ROOT/name).read_bytes()
    bound={name:hashlib.sha256(content).hexdigest() for name,content in payload.items()}
    manifest={'run_id':cfg['run_id'],'study_phase':'C4','files':bound,'models':provenance,'new_training_updates':0,
              'episodes':3*len(confirmation_plan(cfg)),'quality_actions_without_resume':42,'maximum_job_wall_seconds':cfg['max_wall_seconds'],'validation':validation}
    folder=Path(output_dir);folder.mkdir(parents=True,exist_ok=True);archive=folder/(cfg['run_id']+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name,content in sorted(payload.items()):z.writestr('v2/'+name,content)
        z.writestr('v2/'+cfg.get('release_manifest','grpo_c4_release_manifest.json'),json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(archive) as z:
        expected={'v2/'+n for n in bound}|{'v2/'+cfg.get('release_manifest','grpo_c4_release_manifest.json')}
        if z.testzip() is not None or set(z.namelist())!=expected or len(z.namelist())!=len(expected):raise ValueError('C4 release coverage or CRC differs')
        for name,sha in bound.items():
            if hashlib.sha256(z.read('v2/'+name)).hexdigest()!=sha:raise ValueError('C4 archived file hash differs')
    verify_files(ROOT,files)
    receipt={'run_id':cfg['run_id'],'archive':archive.name,'sha256':file_hash(archive),'bytes':archive.stat().st_size,'entries':len(bound)+1,'bound_files':len(bound),'crc_and_hashes_passed':True,'models':ADAPTERS}
    (folder/(cfg['run_id']+'-release.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--c3-review-zip',type=Path,required=True);a=p.parse_args();build(a.output_dir,a.c3_review_zip,a.config)
