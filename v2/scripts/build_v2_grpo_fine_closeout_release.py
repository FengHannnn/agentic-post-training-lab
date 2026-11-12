"""Validate native fine closeout inputs or build the hash-bound GPU release."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import zipfile

from tqdm import tqdm

PREPARATION = 'reports/grpo-fine-closeout-credit-preparation-v1-20261112.md'


def validate(history=None):
    from grpo_common import ROOT, read, validate_config, verify_files
    from apt_v2 import grpo_fine_closeout as study
    from run_v2_grpo_fine_closeout import branch_config
    import sqlite3
    from transformers import AutoTokenizer
    from apt_v2.dpo import corrective_action
    from apt_v2.grpo import initial_state
    from apt_v2.grpo_credit import CLOSEOUT, LOCAL, compute_credit
    from apt_v2.grpo_curriculum import build_context
    from apt_v2.grpo_evaluation import evaluation_plan
    from apt_v2.grpo_repair_study import frozen_inputs, training_tasks
    if sqlite3.sqlite_version!='3.54.1':
        raise ValueError('SQLite 3.54.1 required')
    cfg=read(ROOT/study.CONFIG)
    study.validate_master(cfg)
    baseline, candidate = [branch_config(n,list(study.STATES)) for n in study.BRANCHES]
    if {k for k in baseline if baseline[k]!=candidate[k]}!={'run_id','credit_method'}:
        raise ValueError('training branches differ beyond identity and credit')
    token_dir=ROOT/'artifacts/inputs/grpo-sql-closeout-20261112/tokenizer'
    identity=read(ROOT/baseline['initial_dir']/'model_identity.json')
    verify_files(token_dir,{n:h for n,h in identity['files_sha256'].items() if not n.endswith('.safetensors')})
    tok=AutoTokenizer.from_pretrained(token_dir,local_files_only=True,trust_remote_code=False)
    panels={}
    for name in tqdm(('local','confirm-closeout'),desc='Replay expert panels',unit='panel',file=sys.stdout,disable=False):
        branch=branch_config(name,list(study.STATES))
        validate_config(branch)
        proof=frozen_inputs(branch)
        panels[branch['evaluation_panel']]={'standard_episodes':len(evaluation_plan(branch)),
                    'targeted_episodes':len(proof['targeted_evaluation_plan.json']),
                    'expert_proofs':len(proof['curriculum_expert_proof.json'])}
    maxima={'prompt':0,'response':0}
    tasks={t.id:t for t in training_tasks()}
    plan=study.diagnostic_plan(cfg)
    for spec in tqdm(plan,desc='Encode closeout continuations',unit='start',file=sys.stdout,disable=False):
        env,messages,*_=initial_state(tasks[spec['task_id']],spec['scenario'],context_builder=build_context)
        try:
            while not env.done:
                action=corrective_action(env)
                prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                response=tok(json.dumps(action),add_special_tokens=False)['input_ids']
                if len(prompt)+len(response)+1>8192 or len(response)+1>256:
                    raise ValueError('expert continuation exceeds token limits')
                maxima['prompt']=max(maxima['prompt'],len(prompt))
                maxima['response']=max(maxima['response'],len(response)+1)
                row=env.step(action)
                messages.extend([{'role':'assistant','content':json.dumps(action)},
                                 {'role':'user','content':json.dumps({'observation':row['observation']})}])
            if not env.success:
                raise ValueError('expert closeout failed')
        finally:
            env.close()
    result={'passed':True,'panels':panels,'preflight_groups':len(plan),'max_groups_per_arm':study.GROUPS,
            'max_wall_seconds':study.CAP,'native_tokens':maxima,'sqlite':sqlite3.sqlite_version}
    if history:
        counts={'groups':0,'local_eligible':0,'closeout_eligible':0,'zero_to_positive_check':0,
                'zero_to_positive_submit':0}
        for index in tqdm(range(46),desc='Replay saved training credit',unit='group',file=sys.stdout,disable=False):
            g=read(history/f'group-{index:04}.json')
            old=compute_credit(g['episodes'],method=LOCAL)
            new=compute_credit(g['episodes'],method=CLOSEOUT)
            if old!=g['credit']:
                raise ValueError('historical local credit replay differs')
            counts['groups']+=1
            counts['local_eligible']+=old['eligible']
            counts['closeout_eligible']+=new['eligible']
            for i,e in enumerate(g['episodes']):
                for t,row in enumerate(e['turns']):
                    if old['turn_advantages'][i][t]!=0 or new['turn_advantages'][i][t]<=0:
                        continue
                    tool=row['action'].get('tool')
                    if tool in ('check_result','submit'):
                        counts['zero_to_positive_check' if tool=='check_result' else 'zero_to_positive_submit']+=1
        if counts!={'groups':46,'local_eligible':37,'closeout_eligible':37,
                    'zero_to_positive_check':32,'zero_to_positive_submit':32}:
            raise ValueError('historical credit comparison differs')
        result['historical_replay']=counts
    return result


def build(output):
    from grpo_common import ROOT, atomic_json, file_hash, read, verify_files
    from apt_v2 import grpo_fine_closeout as study
    from run_v2_grpo import source_files
    cfg=read(ROOT/study.CONFIG)
    study.validate_master(cfg)
    validation=read(ROOT/study.VALIDATION)
    if not validation.get('passed') or not validation.get('regression',{}).get('passed'):
        raise ValueError('native validation and relevant regression required')
    files=source_files()
    for folder in (ROOT/'artifacts/inputs',ROOT/'tests/fixtures'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.suffix not in ('.pt','.pyc') and '__pycache__' not in p.parts:
                files[p.relative_to(ROOT).as_posix()]=file_hash(p)
    for n in (study.CONTRACT,study.VALIDATION,PREPARATION,'README.md','PROJECT_REQUIREMENTS.md'):
        files[n]=file_hash(ROOT/n)
    manifest={'run_id':study.RUN_ID,'wall_seconds':study.CAP,'files':files,'local_validation':validation}
    content=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
    output.mkdir(parents=True,exist_ok=True)
    archive=output/(study.RUN_ID+'-run.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in tqdm(sorted(files),desc='Package fine closeout',unit='file',file=sys.stdout,disable=False):
            z.write(ROOT/name,'v2/'+name)
        z.writestr('v2/'+study.MANIFEST,content)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len({n.casefold() for n in names}) or set(names)!={'v2/'+n for n in [*files,study.MANIFEST]} or z.testzip() is not None:
            raise ValueError('release path, CRC or coverage differs')
        for name,digest in tqdm(files.items(),desc='Verify release hashes',unit='file',file=sys.stdout,disable=False):
            if hashlib.sha256(z.read('v2/'+name)).hexdigest()!=digest:
                raise ValueError('release hash differs: '+name)
    verify_files(ROOT,files)
    (ROOT/study.MANIFEST).write_bytes(content)
    receipt={'run_id':study.RUN_ID,'archive':archive.name,'bytes':archive.stat().st_size,
             'sha256':file_hash(archive),'entries':len(files)+1,'bound_files':len(files),'max_wall_seconds':study.CAP}
    atomic_json(output/(study.RUN_ID+'-release.json'),receipt)
    print(json.dumps(receipt,indent=2))
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    args_mode=parser.add_mutually_exclusive_group(required=True)
    args_mode.add_argument('--output-dir',type=Path)
    args_mode.add_argument('--validate',type=Path)
    parser.add_argument('--sqlite-dll',type=Path)
    parser.add_argument('--historical-groups',type=Path)
    args=parser.parse_args()
    if args.validate:
        if args.sqlite_dll:
            sqlite_runtime=ctypes.WinDLL(str(args.sqlite_dll.resolve()))
        from grpo_common import atomic_json
        atomic_json(args.validate,validate(args.historical_groups))
    else:
        build(args.output_dir)
