"""Before/after complete tasks and held-out development recovery actions."""
import sys,os,json,math
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation
from apt_v2.dpo_recovery import SCENARIOS,build_context
from apt_v2.dpo_recovery_training import recovery_result,summarize_recovery

def checkpoints(cfg,output):
    if cfg.get('algorithm')=='chosen_only_sft':
        return {'before':ROOT/'artifacts'/cfg['initial_run']/'final','dpo':ROOT/'artifacts'/cfg['comparison_run']/'final','after':output/'final'}
    return {'before':ROOT/'artifacts'/cfg['initial_run']/'final','after':output/'final'}

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    config_name=cfg.get('config_name','dpo-recovery-train-v1.json')
    if config_name not in ('dpo-recovery-train-v1.json','dpo-recovery-train-v2.json','chosen-only-sft-v1.json') or cfg!=read(ROOT/'configs'/config_name):raise ValueError('fixed recovery training config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,binding['provenance'])
    for branch,p in checkpoints(cfg,output).items():
        complete=read(p/'complete.json');verify_files(p/'adapter',complete['adapter_sha256'])
        if branch=='before' and file_hash(p/'adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('initial DPO changed')
        if branch=='dpo' and file_hash(p/'adapter/adapter_model.safetensors')!=cfg['comparison_adapter_sha256']:raise ValueError('comparison DPO changed')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',read(output/'model_identity.json')['files_sha256'])
    return cfg

def greedy(output):
    cfg=verify_inputs(output)
    from evaluate_v2_dpo import evaluate
    evaluate(output,checkpoints(cfg,output));verify_inputs(output)

def sample(output):
    cfg=verify_inputs(output)
    from evaluate_v2_dpo import sample as evaluate_sample
    evaluate_sample(output,split='development',checkpoints_override=checkpoints(cfg,output));verify_inputs(output)

def recovery(output):
    cfg=verify_inputs(output);ev=cfg['eval'];rc=cfg['recovery_eval']
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    tasks=[t for t in make_tasks() if t.split=='development']
    if rc['split']!='development' or len(tasks)!=rc['task_count']:raise ValueError('held-out development recovery tasks required')
    contexts=[build_context(t,s,allowed_split='development') for t in tasks for s in SCENARIOS];tm={t.id:t for t in tasks}
    with (output/'recovery_contexts.jsonl').open('x',encoding='utf8') as f:
        for c in contexts:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],max_num_seqs=1,
        max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=rc['seed'],enable_lora=True,max_lora_rank=8,max_loras=max(2,len(checkpoints(cfg,output))))
    summaries={}
    try:
        for bi,(branch,path) in enumerate(checkpoints(cfg,output).items(),1):
            request=LoRARequest(branch,bi,str(path/'adapter'));rows=[]
            with (output/(branch+'-recovery-actions.jsonl')).open('x',encoding='utf8') as stream:
                for c in contexts:
                    prompt=tok.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)
                    if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('recovery context budget exceeded')
                    for mode in ('greedy','sample'):
                        for ri in range(1 if mode=='greedy' else rc['samples_per_context']):
                            seed=rc['seed']+[t.id for t in tasks].index(c['task_id'])*10000+list(SCENARIOS).index(c['scenario'])*100+ri
                            params=SamplingParams(n=1,temperature=0 if mode=='greedy' else rc['temperature'],top_p=1,top_k=-1,
                                max_tokens=ev['max_new_tokens'],logprobs=1,stop_token_ids=[tok.eos_token_id],detokenize=False,seed=seed)
                            native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                            if list(native.prompt_token_ids)!=prompt:raise ValueError('native recovery prompt differs')
                            completion=native.outputs[0];ids=list(completion.token_ids)
                            if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native recovery probability alignment')
                            raw=[p[t].logprob for t,p in zip(ids,completion.logprobs)]
                            if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite native recovery probabilities required')
                            generation=asdict(Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                'greedy' if mode=='greedy' else 'dpo_sample',list(prompt),ids,raw,None,completion.finish_reason))
                            result=recovery_result(tm[c['task_id']],c,generation,tok,generation['mode'])
                            row={'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':'development','context_id':c['context_id'],
                                'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,'temperature':0 if mode=='greedy' else rc['temperature'],
                                'generation':generation,'result':result}
                            rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    print(json.dumps({'branch':branch,'context_id':c['context_id'],'actions':len(rows)}),flush=True)
            summaries[branch]={mode:{'overall':summarize_recovery([r for r in rows if r['mode']==mode]),
                'by_scenario':{s:summarize_recovery([r for r in rows if r['mode']==mode and r['scenario']==s]) for s in SCENARIOS}} for mode in ('greedy','sample')}
            summaries[branch]['actions_sha256']=file_hash(output/(branch+'-recovery-actions.jsonl'))
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'recovery_comparison.json',summaries);verify_inputs(output)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['greedy','sample','recovery'],required=True)
    a=p.parse_args();globals()[a.mode](a.output)
