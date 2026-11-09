"""Paired SFT/DPO greedy and fresh-seed stochastic evaluation."""
import sys,os,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.runtime import Generation,run_episode
from apt_v2.tasks import make_tasks
from apt_v2.sft_diagnostics import summarize_episodes

def evaluate(output,checkpoints_override=None):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    cfg=read(output/'config.json');ev=cfg['eval'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    identity=read(output/'model_identity.json');verify_files(folder,identity['files_sha256'])
    checkpoints=checkpoints_override if checkpoints_override is not None else {'sft':ROOT/'artifacts'/cfg['parent_run']/'final','dpo':output/'final'}
    for checkpoint in checkpoints.values():verify_files(checkpoint/'adapter',read(checkpoint/'complete.json')['adapter_sha256'])
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    import shutil
    for name in identity['files_sha256']:
        if name.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,p)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,
        max_model_len=ev['max_model_len'],max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],
        gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=ev['seed'],enable_lora=True,max_lora_rank=8,max_loras=max(2,len(checkpoints)))
    params=SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=ev['max_new_tokens'],
        stop_token_ids=[tok.eos_token_id],detokenize=False,logprobs=1)
    summaries={}
    try:
        for index,(branch,checkpoint) in enumerate(checkpoints.items(),1):
            request=LoRARequest(branch,index,str(checkpoint/'adapter'))
            def policy(messages):
                prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('context budget exceeded')
                native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                if list(native.prompt_token_ids)!=prompt:raise ValueError('native prompt differs')
                completion=native.outputs[0];ids=list(completion.token_ids)
                if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native logprob alignment')
                raw=[row[token].logprob for token,row in zip(ids,completion.logprobs)]
                if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite native log probabilities required')
                return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                    'greedy',prompt,ids,raw,None,completion.finish_reason)
            summaries[branch]={}
            for split,count in ev['splits'].items():
                tasks=[t for t in make_tasks() if t.split==split]
                if len(tasks)!=count:raise ValueError('fixed split task count differs: '+split)
                episodes=[]
                with (output/(branch+'-'+split+'-episodes.jsonl')).open('x',encoding='utf8') as stream:
                    for task in tasks:
                        try:episode=run_episode(task,policy)
                        except Exception as exc:
                            if hasattr(exc,'episode_evidence'):write(output/('partial-'+branch+'-'+split+'.json'),exc.episode_evidence)
                            raise
                        episode['model_branch']=branch;episodes.append(episode)
                        stream.write(json.dumps(episode,ensure_ascii=False)+'\n');stream.flush()
                        print(json.dumps({'model_branch':branch,'split':split,'task':task.id,'success':episode['success']}),flush=True)
                summary=summarize_episodes(episodes);summary['episodes_sha256']=file_hash(output/(branch+'-'+split+'-episodes.jsonl'))
                summaries[branch][split]=summary;write(output/(branch+'-'+split+'-summary.json'),summary)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'comparison_summary.json',summaries)


def sample(output,split='train',checkpoints_override=None):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    cfg=read(output/'config.json');ev=cfg['eval'];sample_cfg=cfg['sample_eval'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(folder,read(output/'model_identity.json')['files_sha256'])
    checkpoints=checkpoints_override if checkpoints_override is not None else {'sft':ROOT/'artifacts'/cfg['parent_run']/'final','dpo':output/'final'}
    for path in checkpoints.values():verify_files(path/'adapter',read(path/'complete.json')['adapter_sha256'])
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,
        max_model_len=ev['max_model_len'],max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],
        gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=sample_cfg['seed'],enable_lora=True,max_lora_rank=8,max_loras=max(2,len(checkpoints)))
    summaries={}
    try:
        for branch_index,(branch,path) in enumerate(checkpoints.items(),1):
            request=LoRARequest(branch,branch_index,str(path/'adapter'));episodes=[]
            with (output/(branch+'-sample-episodes.jsonl')).open('x',encoding='utf8') as stream:
                tasks=[t for t in make_tasks() if t.split==split]
                if len(tasks)!=sample_cfg['task_count']:raise ValueError('fixed sample task count differs')
                for ti,task in enumerate(tasks):
                    for ri in range(sample_cfg['rollouts_per_task']):
                        seed=sample_cfg['seed']+ti*10000+ri*100;call_index=0
                        def policy(messages):
                            nonlocal call_index
                            prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                            if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('context budget exceeded')
                            params=SamplingParams(n=1,temperature=sample_cfg['temperature'],top_p=1,top_k=-1,
                                max_tokens=ev['max_new_tokens'],logprobs=1,stop_token_ids=[tok.eos_token_id],
                                detokenize=False,seed=seed+call_index);call_index+=1
                            native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                            if list(native.prompt_token_ids)!=prompt:raise ValueError('native prompt differs')
                            c=native.outputs[0];ids=list(c.token_ids)
                            if not ids or c.logprobs is None or len(ids)!=len(c.logprobs):raise ValueError('sample probability alignment')
                            raw=[r[t].logprob for t,r in zip(ids,c.logprobs)]
                            if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite sample probabilities required')
                            return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                'dpo_sample',prompt,ids,raw,None,c.finish_reason)
                        try:episode=run_episode(task,policy)
                        except Exception as exc:
                            if hasattr(exc,'episode_evidence'):write(output/('partial-sample-'+branch+'.json'),exc.episode_evidence)
                            raise
                        episode.update(model_branch=branch,rollout_id=f'{task.id}-eval-{branch}-{ri:02}',
                            rollout_index=ri,sample_seed=seed,temperature=sample_cfg['temperature'])
                        for i,row in enumerate(episode['turns']):row['sampling_seed']=seed+i
                        episodes.append(episode);stream.write(json.dumps(episode,ensure_ascii=False)+'\n');stream.flush()
                        print(json.dumps({'model_branch':branch,'task':task.id,'rollout':ri,'success':episode['success']}),flush=True)
            summaries[branch]=summarize_episodes(episodes)
            summaries[branch]['episodes_sha256']=file_hash(output/(branch+'-sample-episodes.jsonl'))
            summaries[branch]['successes_by_task']={t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in tasks}
            write(output/(branch+'-sample-summary.json'),summaries[branch])
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'stochastic_comparison.json',summaries)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['greedy','sample'],default='greedy')
    a=p.parse_args();evaluate(a.output) if a.mode=='greedy' else sample(a.output)
