"""Greedy 1.5B agent baseline: policy actions, tool feedback, repair and verified submission."""
import sys,os,json,math,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,OUTPUT,read,write,verify_files,source_files,file_hash
from apt_v2.tasks import make_tasks
from apt_v2.runtime import run_episode,Generation

def evaluate(output=OUTPUT):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    import torch
    from importlib.metadata import version
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    if not torch.cuda.is_available():raise ValueError('CUDA is required for this model baseline')
    cfg=read(ROOT/'configs/baseline.json');dest=ROOT/'.cache/qwen2.5-1.5b-instruct'
    identity=read(dest/'model_identity.json');verify_files(dest,identity['files_sha256'])
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    tok=AutoTokenizer.from_pretrained(dest,local_files_only=True,trust_remote_code=False)
    params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=cfg['max_new_tokens'],
        logprobs=1,stop_token_ids=[tok.eos_token_id],detokenize=False)
    engine=LLM(model=str(dest),tokenizer=str(dest),dtype=cfg['dtype'],trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'])
    import shutil
    for name in identity['files_sha256']:
        if name.endswith(('.json','.txt','.jinja')):
            target=output/'tokenizer'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(dest/name,target)
    write(output/'model_identity.json',identity)
    episodes=[]
    def policy(messages):
        prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if len(prompt)+cfg['max_new_tokens']>cfg['max_model_len']:raise ValueError('episode context budget exceeded')
        native=engine.generate([{'prompt_token_ids':prompt}],params,use_tqdm=False)[0]
        if list(native.prompt_token_ids)!=prompt or len(native.outputs)!=1:raise ValueError('native prompt mismatch')
        completion=native.outputs[0];ids=list(completion.token_ids)
        if not ids or completion.logprobs is None or len(completion.logprobs)!=len(ids):
            raise ValueError('native token/logprob alignment failed')
        raw=[row[token].logprob for token,row in zip(ids,completion.logprobs)]
        if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('invalid native probabilities')
        if tok.eos_token_id in ids[:-1]:raise ValueError('tokens after EOS')
        return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            'greedy',prompt,ids,raw,None,completion.finish_reason)
    tasks=[t for t in make_tasks() if t.split==cfg['split']][:cfg['task_count']]
    try:
        with (output/'episodes.jsonl').open('x',encoding='utf8') as stream:
            for task in tasks:
                try:episode=run_episode(task,policy,cfg['reward_mode'])
                except Exception as exc:
                    if hasattr(exc,'episode_evidence'):write(output/'partial_episode.json',exc.episode_evidence)
                    raise
                episodes.append(episode);stream.write(json.dumps(episode,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                print(json.dumps({'task_id':task.id,'success':episode['success'],'turns':len(episode['turns'])}),flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    summary={'model_id':cfg['model_id'],'resolved_revision':identity['resolved_revision'],
        'device':torch.cuda.get_device_name(0),'versions':{n:version(n) for n in ('torch','transformers','vllm')},
        'tasks':len(episodes),'successes':sum(e['success'] for e in episodes),
        'fault_tasks':sum(e['fault_mode']!='none' for e in episodes),
        'fault_tasks_recovered':sum(e['recovered'] for e in episodes),
        'total_tool_errors':sum(e['tool_errors'] for e in episodes),
        'total_turns':sum(len(e['turns']) for e in episodes),
        'response_tokens':sum(len(r['generation']['response_token_ids']) for e in episodes for r in e['turns']),
        'sampling_mode':'greedy','training_admitted':False,'optimizer_updates':0,
        'episodes_sha256':file_hash(output/'episodes.jsonl')}
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    write(output/'summary.json',summary)
    return summary

if __name__=='__main__':evaluate()
