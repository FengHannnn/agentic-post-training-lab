"""Evaluate SFT LoRA with the original greedy vLLM baseline settings."""
import sys,json,math,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.runtime import Generation,run_episode,parse_action
from apt_v2.tasks import make_tasks

def evaluate(output):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    cfg=read(output/'config.json')['eval'];dest=ROOT/'.cache/qwen2.5-1.5b-instruct'
    verify_files(dest,read(output/'model_identity.json')['files_sha256'])
    checkpoint=output/'final';verify_files(checkpoint/'adapter',read(checkpoint/'complete.json')['adapter_sha256'])
    tok=AutoTokenizer.from_pretrained(dest,local_files_only=True)
    engine=LLM(model=str(dest),tokenizer=str(dest),dtype='bfloat16',trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'],enable_lora=True,max_lora_rank=8)
    params=SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=cfg['max_new_tokens'],
        stop_token_ids=[tok.eos_token_id],detokenize=False,logprobs=1)
    adapter=LoRARequest('sft-pilot',1,str(checkpoint/'adapter'))
    def policy(messages):
        prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if len(prompt)+cfg['max_new_tokens']>cfg['max_model_len']:raise ValueError('context budget exceeded')
        native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=adapter,use_tqdm=False)[0]
        if list(native.prompt_token_ids)!=prompt:raise ValueError('prompt identity differs')
        completion=native.outputs[0];ids=list(completion.token_ids)
        if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('token alignment')
        raw=[r[t].logprob for t,r in zip(ids,completion.logprobs)]
        if any(not math.isfinite(v) for v in raw):raise ValueError('finite probabilities required')
        return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            'greedy',prompt,ids,raw,None,completion.finish_reason)
    episodes=[]
    try:
        with (output/'evaluation.jsonl').open('x',encoding='utf8') as stream:
            for task in [t for t in make_tasks() if t.split==cfg['split']][:cfg['task_count']]:
                episode=run_episode(task,policy);episodes.append(episode)
                stream.write(json.dumps(episode,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({'task_id':task.id,'success':episode['success']}),flush=True)
    finally:engine.llm_engine.engine_core.shutdown()
    rows=[r for e in episodes for r in e['turns']]
    formatted=0
    for r in rows:
        try:parse_action(r['generation']['text']);formatted+=1
        except (TypeError,ValueError):pass
    write(output/'evaluation_summary.json',{'tasks':len(episodes),'successes':sum(e['success'] for e in episodes),
        'fault_tasks':sum(e['fault_mode']!='none' for e in episodes),'recovered':sum(e['recovered'] for e in episodes),
        'strict_format':formatted,'turns':len(rows),'tool_errors':sum(e['tool_errors'] for e in episodes),
        'verified_metric_events':sum(len(r['new_verified_metrics']) for r in rows),
        'valid_sql':sum(r['action'].get('tool')=='query_sql' and bool(r['observation'].get('verified_metrics')) for r in rows),
        'valid_checks':sum(r['action'].get('tool')=='check_result' and r['observation'].get('passed',False) for r in rows),
        'response_tokens':sum(len(r['generation']['response_token_ids']) for r in rows),
        'episodes_sha256':file_hash(output/'evaluation.jsonl'),
        'baseline':{'successes':0,'tasks':4,'recovered':0,'fault_tasks':2,'strict_format':26,'turns':48},
        'baseline_episodes_sha256':'97d77ba04e68b8b68482abb6113167464199b996f78dbaa3981649082eafb31a'})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);evaluate(p.parse_args().output)
