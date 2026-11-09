"""Paired task sampling and greedy complete recovery at all DPO milestones."""
import sys,os,json
from pathlib import Path
from dpo_investigation_common import ROOT,read,write,load,verify_inputs,checkpoint_path,seed_for
from apt_v2.dpo_investigation import investigation_tasks,investigation_contexts,evaluation_summary
from apt_v2.runtime import run_episode
from apt_v2.matched_dpo import run_recovery
from test_v2_hard_recovery import NativePolicy,shutdown

def evaluate(output):
    cfg=verify_inputs(output)
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';ev=cfg['eval'];tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    capacity=1
    while capacity<len(cfg['checkpoints']):capacity*=2
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],max_num_seqs=1,
        max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=ev['seed'],enable_lora=True,max_lora_rank=8,max_loras=capacity)
    ts=investigation_tasks();cs=investigation_contexts();tm={t.id:t for t in ts};allcomplete=[];allrecovery=[]
    total=len(cfg['checkpoints'])*(len(ts)*(1+cfg['sample_eval']['rollouts_per_task'])+len(cs))
    try:
        with tqdm(total=total,desc='DPO milestone diagnosis',unit='episode',file=sys.stdout,disable=False,dynamic_ncols=True) as progress:
            for bi,(branch,spec) in enumerate(cfg['checkpoints'].items(),1):
                lora=LoRARequest(branch,bi,str(checkpoint_path(spec)/'adapter'))
                for kind,items in [('complete',ts),('recovery',cs)]:
                    with (output/(branch+'-'+kind+'-episodes.jsonl')).open('x',encoding='utf8') as stream:
                        for item in items:
                            c=None if kind=='complete' else item;task=item if c is None else tm[c['task_id']]
                            for mode in (('greedy','sample') if c is None else ('greedy',)):
                                count=1 if mode=='greedy' else cfg['sample_eval']['rollouts_per_task'];temperature=0 if mode=='greedy' else cfg['sample_eval']['temperature']
                                for ri in range(count):
                                    seed=seed_for(cfg,task,mode,ri,c);policy=NativePolicy(engine,tok,lora,cfg,mode,seed,temperature)
                                    try:e=run_episode(task,policy) if c is None else run_recovery(task,c,policy)
                                    except Exception as exc:
                                        if hasattr(exc,'episode_evidence'):write(output/('partial-'+branch+'-'+kind+'.json'),exc.episode_evidence)
                                        raise
                                    e.update(model_branch=branch,mode=mode,sample_seed=seed,temperature=temperature,adapter_sha256=spec['adapter_sha256'])
                                    e['rollout_index' if c is None else 'sample_index']=ri
                                    for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                                    stream.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                                    (allcomplete if c is None else allrecovery).append(e)
                                    progress.set_postfix(branch=branch,kind=kind,success=e['success'],refresh=False);progress.update(1)
    finally:shutdown(engine)
    write(output/'comparison_summary.json',evaluation_summary(allcomplete,allrecovery,cfg['checkpoints']))
    verify_inputs(output)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);evaluate(p.parse_args().output)
