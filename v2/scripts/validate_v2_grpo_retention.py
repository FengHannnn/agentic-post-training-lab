"""Freeze all retention actions after execution, native-token and syntax checks."""
import argparse,ctypes,json,sys,time
from pathlib import Path


def validate(tokenizer_dir,output):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from dataclasses import asdict
    from grpo_common import ROOT,read,atomic_json,file_hash,validate_config
    from apt_v2.grpo_retention_study import validate_master,validate_prior,supervision_dataset,order
    from apt_v2.grpo_repair_study import training_tasks,development_tasks,assert_isolation
    from apt_v2.grpo import digest
    from apt_v2.sft import encode_example
    started=time.monotonic();cfg=read(ROOT/'configs/grpo-sql-retention-study-v1.json');validate_master(cfg)
    validate_prior(read(ROOT/cfg['prior_evidence']),cfg);branch=read(ROOT/cfg['branch_template']);validate_config(branch)
    identity=read(ROOT/branch['initial_dir']/'model_identity.json')
    for n,sha in identity['files_sha256'].items():
        if not n.endswith('.safetensors') and file_hash(tokenizer_dir/n)!=sha:raise ValueError('tokenizer identity differs')
    tok=AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    files=[p for folder in ('src','scripts','tests') for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    for p in tqdm(files,desc='Validate retention syntax',unit='file',file=sys.stdout,disable=False):compile(p.read_bytes(),str(p),'exec')
    data=supervision_dataset();max_prompt=max_response=0
    for e in tqdm(data,desc='Validate retention masks',unit='action',file=sys.stdout,disable=False):
        sample=encode_example(tok,e,8192);max_prompt=max(max_prompt,sample['prompt_length']);max_response=max(max_response,sample['response_length'])
        if sample['response_length']>256 or sample['input_ids'][-1]!=tok.eos_token_id:
            raise ValueError('native response budget or EOS differs')
    folder=(ROOT/cfg['sft_examples']).parent
    atomic_json(ROOT/cfg['sft_examples'],data);atomic_json(folder/'sft-order.json',order(cfg))
    atomic_json(folder/'train-tasks.json',[asdict(t) for t in training_tasks()])
    atomic_json(folder/'development-tasks.json',[asdict(t) for t in development_tasks()])
    result={'passed':True,'syntax_files':len(files),'sft_actions':60,'target_actions':12,'maintenance_actions':48,
            'maintenance_tasks':12,'maintenance_states_per_task':4,'sft_updates':15,'action_exposures':60,
            'learning_rate':2e-6,'maximum_prompt_tokens':max_prompt,'maximum_response_tokens':max_response,
            'context_limit':8192,'response_limit':256,'dataset_sha256':digest(data),'isolation':assert_isolation(),
            'current_job_budget_seconds':16000,'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],
            'combined_maximum_seconds':16000+cfg['prior_elapsed_seconds'],'elapsed_seconds':time.monotonic()-started}
    atomic_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',type=Path,required=True);p.add_argument('--sqlite-dll',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.sqlite_dll:sqlite_runtime=ctypes.WinDLL(str(a.sqlite_dll.resolve()))
    import sqlite3
    if sqlite3.sqlite_version!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    root=Path(__file__).resolve().parents[1]
    validate(a.tokenizer_dir,a.output or root/'reports/grpo-sql-retention-validation-v1-20261110.json')
