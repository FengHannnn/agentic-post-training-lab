"""Four-model paired development evaluation with complete recovery continuations."""
import os,sys,json
from pathlib import Path
from dataclasses import asdict
from dpo_uniform_common import ROOT,read,write,load,tasks,contexts,checkpoints,seed_for,verify_preflight,result_summary
from common import file_hash,verify_files
from apt_v2.runtime import run_episode
from apt_v2.matched_dpo import run_recovery
from apt_v2.recovery_diagnostics import diagnose_action
from test_v2_hard_recovery import NativePolicy,shutdown

def verify_inputs(output):
    cfg=verify_preflight(output);binding=read(output/'input_binding.json')
    if binding['baseline_adapter_sha256']!=cfg['adapter_sha256'] or binding['config_sha256']!=file_hash(output/'config.json') or binding['weights_sha256']!=file_hash(output/'weights.json') or file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('training binding differs')
    verify_files(ROOT,binding['provenance'])
    for name in (*[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']],'final'):
        p=output/name;verify_files(p/'adapter',read(p/'complete.json')['adapter_sha256'])
        if read(p/'binding.json')!=binding:raise ValueError('checkpoint binding differs')
    return cfg

def evaluate(output,stage):
    cfg=verify_inputs(output)
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';ev=cfg['eval'];tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],
        max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,
        enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=ev['seed'],enable_lora=True,max_lora_rank=8,max_loras=4)
    ts=tasks();cs=contexts();tm={t.id:t for t in ts};rc=cfg['recovery_eval'];sc=cfg['sample_eval']
    total=len(cfg['evaluation_branches'])*(len(ts)*(1+sc['rollouts_per_task']) if stage=='complete' else len(cs)*(1+rc['samples_per_context']))
    try:
        with tqdm(total=total,desc='Matched DPO '+stage,unit='episode',file=sys.stdout,disable=False,dynamic_ncols=True) as progress:
            for bi,(branch,p) in enumerate(checkpoints(cfg,output).items(),1):
                lora=LoRARequest(branch,bi,str(p/'adapter'))
                with (output/(branch+'-'+stage+'-episodes.jsonl')).open('x',encoding='utf8') as stream:
                    af=(output/(branch+'-first-actions.jsonl')).open('x',encoding='utf8') if stage=='recovery' else None
                    try:
                        for item in (ts if stage=='complete' else cs):
                            context=None if stage=='complete' else item;task=item if context is None else tm[item['task_id']]
                            for mode in ('greedy','sample'):
                                count=1 if mode=='greedy' else (sc['rollouts_per_task'] if stage=='complete' else rc['samples_per_context'])
                                temperature=0 if mode=='greedy' else (sc['temperature'] if stage=='complete' else rc['temperature'])
                                for ri in range(count):
                                    seed=seed_for(cfg,task,mode,ri,context);policy=NativePolicy(engine,tok,lora,cfg,mode,seed,temperature)
                                    try:e=run_episode(task,policy) if context is None else run_recovery(task,context,policy)
                                    except Exception as exc:
                                        if hasattr(exc,'episode_evidence'):write(output/('partial-'+branch+'-'+stage+'.json'),exc.episode_evidence)
                                        raise
                                    e.update(model_branch=branch,mode=mode,sample_seed=seed,temperature=temperature)
                                    e['rollout_index' if stage=='complete' else 'sample_index']=ri
                                    for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                                    stream.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                                    if context is not None:
                                        g=e['turns'][0]['generation'];actual,d=diagnose_action(task,context,g['text'])
                                        action={'model_branch':branch,'mode':mode,'task_id':task.id,'family_id':task.family_id,'split':task.split,
                                            'context_id':context['context_id'],'scenario':context['scenario'],'sample_index':ri,'sample_seed':seed,
                                            'temperature':temperature,'generation':g,'row':actual,'diagnostic':d}
                                        af.write(json.dumps(action,ensure_ascii=False,allow_nan=False)+'\n');af.flush()
                                    progress.set_postfix(branch=branch,mode=mode,success=e['success'],refresh=False);progress.update(1)
                    finally:
                        if af is not None:af.close()
    finally:shutdown(engine)
    if stage=='recovery':
        complete=[];recovery=[];actions=[]
        for b in cfg['evaluation_branches']:
            complete.extend(load(output/(b+'-complete-episodes.jsonl')));recovery.extend(load(output/(b+'-recovery-episodes.jsonl')));actions.extend(load(output/(b+'-first-actions.jsonl')))
        write(output/'comparison_summary.json',result_summary(complete,recovery,actions,cfg))
    verify_inputs(output)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['complete','recovery'],required=True)
    a=p.parse_args();evaluate(a.output,a.stage)
