"""Before/final tasks and native recovery probes at every epoch checkpoint."""
import sys,os,json,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.dpo_recovery_training import validate_generation
from apt_v2.recovery_diagnostics import diagnostic_contexts,diagnose_action,diagnostic_summary,summarize
from diagnose_v2_recovery_contexts import sampling_seed,selected_tasks

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]

def checkpoints(cfg,output,intermediate=False):
    if intermediate:return {f'checkpoint-{step:04}':output/f'checkpoint-{step:04}' for step in cfg['checkpoint_steps']}
    return {'before':ROOT/'artifacts'/cfg['initial_run']/'final','after':output/'final'}

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs/hard-recovery-sft-weighted-v1.json'):raise ValueError('fixed weighted recovery evaluation required')
    if read(output/'run_manifest.json')['source_sha256']!=binding['source_sha256']:raise ValueError('training source binding differs')
    verify_files(ROOT,binding['source_sha256']);verify_files(ROOT,binding['provenance'])
    if file_hash(output/'config.json')!=binding['config_sha256'] or file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256'] or file_hash(output/'weights.json')!=binding['weights_sha256']:
        raise ValueError('weighted recovery inputs differ')
    if file_hash(output/'contexts.jsonl')!=binding['contexts_sha256'] or file_hash(output/'context_membership.jsonl')!=binding['membership_sha256']:raise ValueError('recovery contexts differ')
    for branch,p in {**checkpoints(cfg,output),**checkpoints(cfg,output,True)}.items():
        c=read(p/'complete.json');verify_files(p/'adapter',c['adapter_sha256'])
        if branch=='before' and file_hash(p/'adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('initial weighted SFT model differs')
        if branch!='before' and read(p/'binding.json')!=binding:raise ValueError('checkpoint input binding differs')
    if read(output/'model_identity.json')!=read(ROOT/'artifacts'/cfg['initial_run']/'model_identity.json'):raise ValueError('parent base identity differs')
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

def probe(output,intermediate=False):
    cfg=verify_inputs(output);tasks=selected_tasks(cfg);tm={t.id:t for t in tasks};contexts=load(output/'contexts.jsonl')
    if contexts!=diagnostic_contexts(tasks):raise ValueError('fixed recovery context manifest differs')
    if intermediate:contexts=[c for c in contexts if c['split']=='train']
    branches=checkpoints(cfg,output,intermediate);ev=cfg['eval'];rc=cfg['recovery_eval'];modes=('greedy',) if intermediate else ('greedy','sample')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    capacity=2
    while capacity<len(branches):capacity*=2
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],
        max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,
        enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=rc['seed'],enable_lora=True,max_lora_rank=8,max_loras=capacity)
    rows=[]
    try:
        total=len(branches)*len(contexts)*(1 if intermediate else 1+rc['samples_per_context'])
        with tqdm(total=total,desc='Checkpoint recovery' if intermediate else 'Before/final recovery',unit='action',file=sys.stdout,disable=False,dynamic_ncols=True,mininterval=.3) as progress:
            for bi,(branch,p) in enumerate(branches.items(),1):
                request=LoRARequest(branch,bi,str(p/'adapter'))
                with (output/(branch+'-actions.jsonl')).open('x',encoding='utf8') as stream:
                    for c in contexts:
                        prompt=tok.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)
                        if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('recovery prompt budget exceeded')
                        for mode in modes:
                            for ri in range(1 if mode=='greedy' else rc['samples_per_context']):
                                seed=sampling_seed(cfg,tasks,c,ri);temperature=0 if mode=='greedy' else rc['temperature']
                                params=SamplingParams(n=1,temperature=temperature,top_p=1,top_k=-1,max_tokens=ev['max_new_tokens'],logprobs=1,
                                    stop_token_ids=[tok.eos_token_id],detokenize=False,seed=seed)
                                native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                                if list(native.prompt_token_ids)!=list(prompt):raise ValueError('native recovery prompt differs')
                                completion=native.outputs[0];ids=list(completion.token_ids)
                                if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native recovery alignment required')
                                raw=[p[t].logprob for t,p in zip(ids,completion.logprobs)]
                                g=asdict(Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                    'greedy' if mode=='greedy' else 'dpo_sample',list(prompt),ids,raw,None,completion.finish_reason))
                                validate_generation(tok,c['messages'],g,g['mode'])
                                actual,d=diagnose_action(tm[c['task_id']],c,g['text'])
                                row={'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':c['split'],'context_id':c['context_id'],
                                    'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,'temperature':temperature,
                                    'generation':g,'row':actual,'diagnostic':d}
                                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                                progress.set_postfix(branch=branch,split=c['split'],scenario=c['scenario'],refresh=False);progress.update(1)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    if intermediate:
        write(output/'checkpoint_diagnostics.json',{b:{'overall':summarize([r for r in rows if r['model_branch']==b]),
            'by_scenario':{s:summarize([r for r in rows if r['model_branch']==b and r['scenario']==s]) for s in SCENARIOS},
            'actions_sha256':file_hash(output/(b+'-actions.jsonl'))} for b in branches})
    else:write(output/'diagnostic_summary.json',diagnostic_summary(rows,branches))
    verify_inputs(output)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['greedy','sample','checkpoints','recovery'],required=True)
    a=p.parse_args()
    if a.mode=='checkpoints':probe(a.output,True)
    elif a.mode=='recovery':probe(a.output,False)
    else:globals()[a.mode](a.output)
