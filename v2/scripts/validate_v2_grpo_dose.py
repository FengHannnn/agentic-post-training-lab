"""Freeze identical supervision content and dose-only multiplicities with native replay checks."""
import argparse,ctypes,json,sys,time,shutil
from pathlib import Path
from collections import Counter


def validate(tokenizer_dir,output):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from grpo_common import ROOT,read,atomic_json,file_hash,validate_config
    from grpo_repair_worker import verify_supervision,sft_order
    from apt_v2.grpo_dose_study import validate_master,validate_prior,warm_config,DOSES
    from apt_v2.grpo_retention_study import supervision_dataset
    from apt_v2.grpo_warmup_probe import make_plan,execute,productive,variants
    from apt_v2.grpo import digest
    from apt_v2.sft import encode_example
    started=time.monotonic();cfg=read(ROOT/'configs/grpo-sql-dose-study-v1.json');validate_master(cfg)
    folder=ROOT/cfg['input_dir'];validate_prior(read(folder/'prior-evidence.json'),cfg)
    branch=read(ROOT/cfg['branch_template']);validate_config(branch)
    identity=read(ROOT/branch['initial_dir']/'model_identity.json');(folder/'tokenizer').mkdir(exist_ok=True)
    for n,sha in identity['files_sha256'].items():
        if not n.endswith('.safetensors'):
            if file_hash(tokenizer_dir/n)!=sha:raise ValueError('frozen tokenizer differs')
            shutil.copyfile(tokenizer_dir/n,folder/'tokenizer'/n)
    tok=AutoTokenizer.from_pretrained(folder/'tokenizer',local_files_only=True,trust_remote_code=False)
    data=supervision_dataset();verify_supervision(data,expected_count=60)
    if digest(data)!='0e53fafba730aa7ad815457fc60d0a973e77a015774ea5c8e583abb38ba36dc2':
        raise ValueError('dose study must preserve retention supervision content')
    atomic_json(ROOT/cfg['sft_examples'],data);plan=make_plan(data);atomic_json(folder/'probe-plan.json',plan)
    maximum_prompt=maximum_response=0
    for e in tqdm(data,desc='Validate dose supervision',unit='action',file=sys.stdout,disable=False):
        encoded=encode_example(tok,e,8192);maximum_prompt=max(maximum_prompt,encoded['prompt_length']);maximum_response=max(maximum_response,encoded['response_length'])
        if encoded['response_length']>256:raise ValueError('native response budget exceeded')
    doses={}
    for r in tqdm(DOSES,desc='Freeze target multiplicities',unit='dose',file=sys.stdout,disable=False):
        wc=warm_config(cfg,r);order=sft_order(wc);counts=Counter(sum(order,[]))
        if any(counts[i]!=(r if e['role']=='target' else 1) for i,e in enumerate(data)):
            raise ValueError('only target multiplicity may change')
        atomic_json(folder/f'order-r{r}.json',order)
        doses[str(r)]={'steps':len(order),'target_exposures':12*r,'maintenance_exposures':48,'total_exposures':sum(counts.values()),
                       'source_sft_updates':513+len(order),'order_sha256':digest(order)}
    for e in tqdm(plan,desc='Replay dose probes',unit='context',file=sys.stdout,disable=False):
        if not productive(e,execute(e,e['response'])):raise ValueError('expert probe action failed')
        if e['kind']=='target' and productive(e,execute(e,variants(tok,e)['omitted_text'])):
            raise ValueError('missing-alias counterfactual must remain unverified')
    files=[p for d in ('src','scripts','tests') for p in (ROOT/d).rglob('*.py') if '__pycache__' not in p.parts]
    for p in tqdm(files,desc='Validate dose syntax',unit='file',file=sys.stdout,disable=False):compile(p.read_bytes(),str(p),'exec')
    result={'passed':True,'syntax_files':len(files),'unique_supervision_actions':60,'target_actions':12,'maintenance_actions':48,
            'maintenance_tasks':12,'fixed_learning_rate':2e-6,'fixed_batch_size':4,'doses':doses,'dataset_sha256':digest(data),
            'probe_contexts_per_model':24,'target_first_action_repeats':2,'maximum_prompt_tokens':maximum_prompt,
            'maximum_response_tokens':maximum_response,'context_limit':8192,'response_limit':256,
            'new_authorized_seconds':21600,'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],
            'combined_maximum_seconds':cfg['total_authorized_seconds'],'elapsed_seconds':time.monotonic()-started}
    atomic_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',type=Path,required=True);p.add_argument('--sqlite-dll',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.sqlite_dll:sqlite_runtime=ctypes.WinDLL(str(a.sqlite_dll.resolve()))
    import sqlite3
    if sqlite3.sqlite_version!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    root=Path(__file__).resolve().parents[1];validate(a.tokenizer_dir,a.output or root/'reports/grpo-sql-dose-validation-v1-20261110.json')
