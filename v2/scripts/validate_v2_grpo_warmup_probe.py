"""Freeze train-only probe inputs after real execution and native-token branch checks."""
import argparse,ctypes,json,sys,time,shutil
from pathlib import Path


def validate(tokenizer_dir,output):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from grpo_common import ROOT,read,atomic_json,file_hash
    from grpo_repair_worker import verify_supervision
    from apt_v2.grpo_retention_study import supervision_dataset
    from apt_v2.grpo_warmup_probe import validate_config,validate_prior,make_plan,execute,productive,variants
    from apt_v2.grpo import digest
    from grpo_warmup_probe_worker import model_identity
    started=time.monotonic();cfg=read(ROOT/'configs/grpo-sql-warmup-probe-v1.json');validate_config(cfg)
    folder=ROOT/cfg['input_dir'];validate_prior(read(folder/'prior-evidence.json'),cfg)
    identity=None
    for name in tqdm(cfg['models'],desc='Verify probe adapters',unit='model',file=sys.stdout,disable=False):
        actual=model_identity(cfg,folder/'models'/name,name)
        if identity is not None and actual!=identity:raise ValueError('three frozen base identities differ')
        identity=actual
    (folder/'tokenizer').mkdir(exist_ok=True)
    for n,sha in identity['files_sha256'].items():
        if not n.endswith('.safetensors'):
            if file_hash(tokenizer_dir/n)!=sha:raise ValueError('probe tokenizer differs')
            shutil.copyfile(tokenizer_dir/n,folder/'tokenizer'/n)
    tok=AutoTokenizer.from_pretrained(folder/'tokenizer',local_files_only=True,trust_remote_code=False)
    data=supervision_dataset();verify_supervision(data,expected_count=60);plan=make_plan(data)
    maximum_prompt=maximum_response=0;branch_records=[]
    from apt_v2.sft import encode_example
    for item in tqdm(plan,desc='Validate train probes',unit='context',file=sys.stdout,disable=False):
        if not productive(item,execute(item,item['response'])):raise ValueError('expert probe action not productive')
        encoded=encode_example(tok,item,8192);maximum_prompt=max(maximum_prompt,encoded['prompt_length'])
        maximum_response=max(maximum_response,encoded['response_length'])
        if encoded['response_length']>256:raise ValueError('greedy expert action exceeds native limit')
        if item['kind']=='target':
            v=variants(tok,item)
            if productive(item,execute(item,v['omitted_text'])):raise ValueError('omitted alias must not be certified')
            if len(v['prompt_ids'])+max(len(v['correct_ids']),len(v['omitted_ids']))+1>8192:
                raise ValueError('forced score exceeds context limit')
            branch_records.append({'task_id':item['task_id'],'branch_index':v['branch_index'],
                                   'correct_branch_token':v['correct_ids'][v['branch_index']],
                                   'omitted_branch_token':v['omitted_ids'][v['branch_index']],
                                   'alias_tokens':len(v['alias_indices'])})
    files=[p for d in ('src','scripts','tests') for p in (ROOT/d).rglob('*.py') if '__pycache__' not in p.parts]
    for p in tqdm(files,desc='Validate probe syntax',unit='file',file=sys.stdout,disable=False):compile(p.read_bytes(),str(p),'exec')
    atomic_json(folder/'sft-examples.json',data);atomic_json(folder/'probe-plan.json',plan)
    result={'passed':True,'syntax_files':len(files),'models':3,'target_train_contexts_per_model':12,
            'maintenance_train_contexts_per_model':12,'target_greedy_repeats':2,'committed_generated_actions':108,
            'committed_forced_score_requests':72,'auxiliary_generated_tokens':72,'supervision_actions_replayed':60,
            'expert_probe_actions_replayed':24,'omitted_alias_counterfactuals_replayed':12,
            'maximum_expert_prompt_tokens':maximum_prompt,'maximum_expert_response_tokens':maximum_response,
            'context_limit':8192,'response_limit':256,'plan_sha256':digest(plan),'branch_records':branch_records,
            'actual_optimizer_updates':0,'current_budget_seconds':1800,'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],
            'combined_maximum_seconds':cfg['prior_elapsed_seconds']+1800,'elapsed_seconds':time.monotonic()-started}
    atomic_json(output,result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer-dir',type=Path,required=True);p.add_argument('--sqlite-dll',type=Path)
    p.add_argument('--output',type=Path);a=p.parse_args()
    if a.sqlite_dll:sqlite_runtime=ctypes.WinDLL(str(a.sqlite_dll.resolve()))
    import sqlite3
    if sqlite3.sqlite_version!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    root=Path(__file__).resolve().parents[1]
    validate(a.tokenizer_dir,a.output or root/'reports/grpo-warmup-probe-validation-v1-20261110.json')
