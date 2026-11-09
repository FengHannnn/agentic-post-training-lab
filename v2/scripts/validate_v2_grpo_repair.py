"""CPU source, expert, supervision and native-token budget validation with visible progress."""
import argparse,ctypes,json,sys,time
from pathlib import Path


def validate(tokenizer_dir,output):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from grpo_common import ROOT,read,atomic_json,file_hash,validate_config
    from apt_v2.grpo_repair_study import (training_tasks,development_tasks,CONDITIONS,STANDARD,context,
                                          assert_isolation,diagnostic_plan,start_spec)
    from apt_v2.grpo import initial_state,digest
    from apt_v2.dpo import corrective_action
    from apt_v2.sft import encode_example
    from grpo_repair_worker import supervision_dataset,sft_order
    started=time.monotonic();cfg=read(ROOT/'configs/grpo-sql-repair-study-v1.json');branch=read(ROOT/cfg['branch_template']);validate_config(branch)
    identity=read(ROOT/branch['initial_dir']/'model_identity.json')
    for name,sha in identity['files_sha256'].items():
        if not name.endswith('.safetensors') and file_hash(tokenizer_dir/name)!=sha:raise ValueError('frozen tokenizer identity differs: '+name)
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    files=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    for p in tqdm(files,desc='Validate source syntax',unit='file',file=sys.stdout,disable=False):compile(p.read_bytes(),str(p),'exec')
    maximum_prompt=maximum_response=proofs=0
    for task in tqdm(training_tasks()+development_tasks(),desc='Expert/token replay',unit='task',file=sys.stdout,disable=False):
        for scenario in (None,*STANDARD,*CONDITIONS):
            env,messages,_,_=initial_state(task,scenario,context_builder=context)
            try:
                while not env.done:
                    action=corrective_action(env);text=json.dumps(action)
                    sample=encode_example(tok,{'messages':messages,'response':text},branch['sampling']['max_model_len'])
                    maximum_prompt=max(maximum_prompt,sample['prompt_length']);maximum_response=max(maximum_response,sample['response_length'])
                    if sample['response_length']>branch['sampling']['max_new_tokens']:raise ValueError('expert response exceeds native limit')
                    row=env.step(action)
                    messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}])
                if not env.success:raise ValueError('expert must complete within original action budget')
                proofs+=1
            finally:env.close()
    examples=supervision_dataset()
    for example in tqdm(examples,desc='Validate response masks',unit='action',file=sys.stdout,disable=False):
        item=encode_example(tok,example,branch['sampling']['max_model_len'])
        if item['response_length']>branch['sampling']['max_new_tokens']:raise ValueError('SFT response budget exceeded')
    atomic_json(ROOT/cfg['sft_examples'],examples)
    frozen=ROOT/cfg['sft_examples'];frozen=frozen.parent
    from dataclasses import asdict
    atomic_json(frozen/'train-tasks.json',[asdict(t) for t in training_tasks()])
    atomic_json(frozen/'development-tasks.json',[asdict(t) for t in development_tasks()])
    atomic_json(frozen/'diagnostic-plan.json',diagnostic_plan(cfg))
    atomic_json(frozen/'sft-order.json',sft_order(cfg))
    result={'passed':True,'scope':'CPU expert replay, source syntax and frozen tokenizer budgets',
            'syntax_files':len(files),'expert_starts':proofs,'isolation':assert_isolation(),
            'maximum_expert_prompt_tokens':maximum_prompt,'maximum_expert_response_tokens':maximum_response,
            'context_limit':8192,'response_limit':256,'sft_actions':24,'sft_updates':24,'action_exposures':96,
            'sft_dataset_sha256':digest(examples),'diagnostic_episodes':len(diagnostic_plan(cfg))*4,
            'elapsed_seconds':time.monotonic()-started,'verifier_version':'sql_semantics_v2'}
    atomic_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',type=Path,required=True);p.add_argument('--sqlite-dll',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.sqlite_dll:sqlite_runtime=ctypes.WinDLL(str(a.sqlite_dll.resolve()))
    import sqlite3
    if sqlite3.sqlite_version!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    root=Path(__file__).resolve().parents[1]
    validate(a.tokenizer_dir,a.output or root/'reports/grpo-sql-repair-validation-v1-20261109.json')
