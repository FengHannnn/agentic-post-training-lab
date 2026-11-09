"""Frozen-SFT stochastic rollout collection and verified action preference audit."""
import sys,os,json,math,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation,run_episode
from apt_v2.dpo import verified_pairs,balanced_pairs,failure_reason
from apt_v2.sft_diagnostics import summarize_episodes


def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def verify_inputs(output):
    cfg=read(output/'config.json');source=ROOT/'artifacts'/cfg['source_run'];binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs/dpo-collect-v1.json'):raise ValueError('collection config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    verify_files(source,binding['source_files']);verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files'])
    return cfg,source

def prepare(output):
    cfg=read(output/'config.json');source=ROOT/'artifacts'/cfg['source_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    complete=read(source/'final/complete.json');verify_files(source/'final/adapter',complete['adapter_sha256'])
    if file_hash(source/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256'] or complete['total_steps']!=126:
        raise ValueError('reviewed frozen 126-step SFT required')
    if cfg['training_enabled'] is not False or cfg['split']!='train':raise ValueError('train-only data collection required')
    identity=read(source/'model_identity.json')
    if identity['resolved_revision']!=cfg['revision']:raise ValueError('fixed model revision required')
    verify_files(folder,identity['files_sha256'])
    examples=[json.loads(l) for l in (source/'examples.jsonl').read_text(encoding='utf8').splitlines()]
    from apt_v2.sql_sft import data_identity
    if any(e['split']!='train' for e in examples) or data_identity(examples)!=read(source/'input_binding.json')['dataset_sha256']:
        raise ValueError('training provenance differs')
    tasks=[t for t in make_tasks() if t.split=='train']
    if len(tasks)!=cfg['task_count']:raise ValueError('fixed train task count differs')
    from dataclasses import asdict
    write(output/'tasks.json',[asdict(t) for t in tasks])
    names=['model_identity.json','examples.jsonl','input_binding.json','final/complete.json']+[
        'final/adapter/'+n for n in complete['adapter_sha256']]
    write(output/'input_binding.json',{'source_files':{n:file_hash(source/n) for n in names},
        'model_files':identity['files_sha256'],'baseline_id':cfg['baseline_id'],'adapter_sha256':cfg['adapter_sha256']})
    write(output/'model_identity.json',identity)
    for name in identity['files_sha256']:
        if name.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/name,p)
    write(output/'prepare_summary.json',{'task_count':len(tasks),'rollout_count':len(tasks)*cfg['rollouts_per_task'],
        'baseline_id':cfg['baseline_id'],'adapter_sha256':cfg['adapter_sha256'],'optimizer_updates':0})

def collect(output):
    cfg,source=verify_inputs(output)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'],enable_lora=True,max_lora_rank=8,max_loras=2)
    request=LoRARequest('frozen-sft-dpo',1,str(source/'final/adapter'))
    episodes=[]
    try:
        with (output/'rollouts.jsonl').open('x',encoding='utf8') as stream:
            for task_index,task in enumerate([t for t in make_tasks() if t.split=='train']):
                for rollout_index in range(cfg['rollouts_per_task']):
                    rollout_seed=cfg['seed']+task_index*10000+rollout_index*100;call_index=0
                    def policy(messages):
                        nonlocal call_index
                        prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                        if len(prompt)+cfg['max_new_tokens']>cfg['max_model_len']:raise ValueError('context budget exceeded')
                        params=SamplingParams(n=1,temperature=cfg['temperature'],top_p=1,top_k=-1,
                            max_tokens=cfg['max_new_tokens'],logprobs=1,stop_token_ids=[tok.eos_token_id],
                            detokenize=False,seed=rollout_seed+call_index)
                        call_index+=1
                        native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                        if list(native.prompt_token_ids)!=prompt:raise ValueError('native prompt differs')
                        completion=native.outputs[0];ids=list(completion.token_ids)
                        if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native probability alignment')
                        raw=[row[token].logprob for token,row in zip(ids,completion.logprobs)]
                        if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite raw log probabilities required')
                        return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                            'dpo_sample',prompt,ids,raw,None,completion.finish_reason)
                    try:episode=run_episode(task,policy)
                    except Exception as exc:
                        if hasattr(exc,'episode_evidence'):write(output/'partial_episode.json',exc.episode_evidence)
                        raise
                    episode.update(rollout_id=f'{task.id}-sample-{rollout_index:02}',rollout_index=rollout_index,
                        sample_seed=rollout_seed,temperature=cfg['temperature'],adapter_sha256=cfg['adapter_sha256'])
                    for index,row in enumerate(episode['turns']):row['sampling_seed']=rollout_seed+index
                    episodes.append(episode);stream.write(json.dumps(episode,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps({'task':task.id,'rollout':rollout_index,'success':episode['success'],'turns':len(episode['turns'])}),flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'rollout_summary.json',{**summarize_episodes(episodes),'sampling_mode':'dpo_sample',
        'temperature':cfg['temperature'],'rollouts_sha256':file_hash(output/'rollouts.jsonl'),'optimizer_updates':0})
    verify_inputs(output)

def build(output):
    from transformers import AutoTokenizer
    from collections import Counter
    cfg,source=verify_inputs(output);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    tasks={t.id:t for t in make_tasks() if t.split=='train'}
    from dataclasses import asdict
    if read(output/'tasks.json')!=[asdict(t) for t in tasks.values()]:raise ValueError('frozen train tasks differ')
    episodes=[json.loads(l) for l in (output/'rollouts.jsonl').read_text(encoding='utf8').splitlines()]
    expected={(t,i) for t in tasks for i in range(cfg['rollouts_per_task'])}
    if len(episodes)!=len(expected) or {(e['task_id'],e['rollout_index']) for e in episodes}!=expected:raise ValueError('complete distinct rollout set required')
    candidates=[];failure_counts=Counter()
    for e in episodes:
        ti=list(tasks).index(e['task_id']);seed=cfg['seed']+ti*10000+e['rollout_index']*100
        if (e['sample_seed']!=seed or e['temperature']!=cfg['temperature'] or e['adapter_sha256']!=cfg['adapter_sha256']
                or e['rollout_id']!=f"{e['task_id']}-sample-{e['rollout_index']:02}"):
            raise ValueError('rollout policy identity differs')
        if any(row['sampling_seed']!=seed+i for i,row in enumerate(e['turns'])):raise ValueError('sampling seed differs')
        candidates+=verified_pairs(tasks[e['task_id']],e,tok,cfg['pair_max_length'],cfg['max_pairs_per_episode'])
        failure_counts.update(failure_reason(row) for row in e['turns'] if failure_reason(row))
    pairs=balanced_pairs(candidates,cfg['max_pairs_per_task'])
    jsonl(output/'pairs.jsonl',pairs)
    summary=summarize_episodes(episodes)
    recorded=read(output/'rollout_summary.json')
    if any(recorded[k]!=v for k,v in summary.items()):raise ValueError('rollout summary differs')
    pair_tasks=len({p['task_id'] for p in pairs})
    write(output/'pair_summary.json',{'raw_candidates':len(candidates),'pairs':len(pairs),'task_coverage':pair_tasks,
        'pairs_by_task':dict(Counter(p['task_id'] for p in pairs)),'pairs_by_reason':dict(Counter(p['reason'] for p in pairs)),
        'observed_failure_actions':dict(failure_counts),'chosen_source':'scripted_counterfactual',
        'rejected_source':'frozen_sft_sample','selection':'deduplicated, capped per episode and task',
        'ready_for_training_review':len(pairs)>=cfg['min_pairs_for_training'] and pair_tasks>=cfg['min_task_coverage_for_training'],
        'pairs_sha256':file_hash(output/'pairs.jsonl'),'optimizer_updates':0})
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,'episodes_replayed':len(episodes),'turns_replayed':sum(len(e['turns']) for e in episodes),
        'counterfactual_pairs_verified':len(candidates),'baseline_unchanged':True,'optimizer_updates':0})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','collect','build'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    globals()[a.stage](a.output)
