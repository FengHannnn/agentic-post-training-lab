"""Bounded 4090D compatibility check for the V2 1.5B agent stack."""
import argparse,contextlib,hashlib,json,math,os,shutil,signal,subprocess,sys,time,traceback,uuid,zipfile
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
EXPECTED={'torch':'2.8.0+cu128','vllm':'0.11.0','transformers':'4.57.6','peft':'0.17.1'}

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def save(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def model_digest(model,frozen=False):
    import torch
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if frozen and p.requires_grad:continue
        h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
        flat=p.detach().reshape(-1)
        for start in range(0,flat.numel(),2**20):
            h.update(flat[start:start+2**20].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def inventory(output):
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for key in ('torch','vllm','transformers','peft','accelerate','safetensors','huggingface-hub'):
        try:versions[key]=version(key)
        except PackageNotFoundError:versions[key]=None
    info={'python':sys.version,'executable':sys.executable,'versions':versions,
        'matches_previous_stack':{k:versions.get(k)==v for k,v in EXPECTED.items()},
        'environment':{k:os.environ.get(k) for k in ('CONDA_PREFIX','CUDA_VISIBLE_DEVICES','VLLM_WORKER_MULTIPROC_METHOD')}}
    try:
        info['nvidia_smi']=subprocess.run(['nvidia-smi',
            '--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader'],
            capture_output=True,text=True,timeout=20).stdout.strip()
    except (OSError,subprocess.TimeoutExpired) as exc:info['nvidia_smi']=str(exc)
    info['disk_free_gib']=shutil.disk_usage(ROOT).free/2**30
    if Path('/proc/meminfo').exists():
        info['system_memory']=Path('/proc/meminfo').read_text().splitlines()[:3]
    save(output/'inventory_packages.json',info)
    import torch
    if not (3,11)<=sys.version_info[:2]<(3,13):raise ValueError('Python 3.11 or 3.12 required')
    missing=[k for k,v in versions.items() if v is None]
    if missing:raise ValueError('missing packages: '+', '.join(missing))
    if not torch.cuda.is_available():raise ValueError('CUDA is not available in the active Python environment')
    torch.cuda.set_device(0)
    free,total=torch.cuda.mem_get_info()
    info.update(device=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability(0)),
        torch_cuda_version=torch.version.cuda,bf16_supported=torch.cuda.is_bf16_supported(),
        total_vram_gib=total/2**30,free_vram_gib=free/2**30)
    save(output/'gpu.json',info)
    if not info['bf16_supported']:raise ValueError('the V2 BF16 profile requires BF16 support')
    torch.manual_seed(20261030)
    a=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    b=torch.randn(128,128,device='cuda',dtype=torch.bfloat16)
    loss=(a@b).float().square().mean();loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(a.grad).all():raise ValueError('CUDA BF16 matrix/backward test failed')
    info.update(cuda_bf16_forward_backward_passed=True,passed=True)
    return info

def model_ready(output,download):
    from common import verify_files
    from prepare_v2 import download as download_model
    cfg=read(ROOT/'configs/baseline.json')
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    if not (folder/'model_identity.json').exists() and not download:
        return {'passed':False,'pending':'model cache is absent; rerun with --download-model',
            'model_downloaded':False}
    identity=download_model() if download else read(folder/'model_identity.json')
    if identity['model_id']!=cfg['model_id'] or not identity['resolved_revision'].startswith(cfg['revision']):
        raise ValueError('model cache revision differs from the V2 pin')
    verify_files(folder,identity['files_sha256'])
    # Check the deterministic agent runtime separately from GPU model behavior.
    from prepare_v2 import preflight
    checks=preflight()
    return {'passed':True,'model_id':identity['model_id'],'resolved_revision':identity['resolved_revision'],
        'cache_files_verified':len(identity['files_sha256']),'agent_reference_tasks_passed':checks['scripted_reference_successes']}

def make_prompt(tok,width):
    ids=tok.apply_chat_template([{'role':'system','content':'Follow the user instruction.'},
        {'role':'user','content':'Reply with OK only.'}],tokenize=True,add_generation_prompt=True)
    if len(ids)>width:raise ValueError('prompt budget is too short')
    filler=tok.encode(' Diagnostic context.',add_special_tokens=False)
    extra=(filler*((width-len(ids))//len(filler)+1))[:width-len(ids)]
    return ids[:-10]+extra+ids[-10:]

def train_probe(output):
    import gc
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,PeftModel,get_peft_model
    cfg=read(ROOT/'configs/baseline.json');folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    torch.manual_seed(cfg['seed'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    def base():
        return AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
    lc=cfg['lora']
    policy=get_peft_model(base(),LoraConfig(task_type='CAUSAL_LM',r=lc['r'],lora_alpha=lc['alpha'],
        lora_dropout=lc['dropout'],target_modules=lc['target_modules'])).eval()
    reference=base().requires_grad_(False)
    named=[(n,p) for n,p in policy.named_parameters() if p.requires_grad]
    if not named or any('lora_' not in n for n,p in named):raise ValueError('only LoRA should be trainable')
    frozen_before=model_digest(policy,True)
    adapter_before={n:p.detach().cpu().clone() for n,p in named}
    response=tok.encode('{"tool":"read_policy","arguments":{}}',add_special_tokens=False)+[tok.eos_token_id]
    prompt=make_prompt(tok,512-len(response))
    from apt_v2.trainer import response_scores
    old=response_scores(policy,prompt,response,False)
    fresh=response_scores(policy,prompt,response,True)
    ref=response_scores(reference,prompt,response,False)
    error=(old-fresh.detach()).abs().max().item()
    if error>1e-5 or not torch.isfinite(ref).all() or not fresh.requires_grad:
        raise ValueError('FP32 same-path probability gate failed')
    optimizer=torch.optim.AdamW([p for n,p in named],lr=1e-5,weight_decay=0.)
    optimizer.zero_grad(set_to_none=True)
    loss=-fresh.mean();loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],1.,error_if_nonfinite=True)
    if norm.item()<=0:raise ValueError('nonzero adapter gradient required')
    optimizer.step()
    changed=sum(not torch.equal(p.detach().cpu(),adapter_before[n]) for n,p in named)
    if changed==0 or model_digest(policy,True)!=frozen_before:
        raise ValueError('adapter/frozen-parameter update checks failed')
    if any(p.grad is not None for p in reference.parameters()):raise ValueError('reference accumulated gradients')
    expected=response_scores(policy,prompt,response,False).cpu()
    adapter=output/'probe_adapter'
    policy.save_pretrained(adapter,save_embedding_layers=False)
    torch.save({'optimizer':optimizer.state_dict(),'step':1,'probe_only':True},output/'probe_optimizer.pt')
    result={'passed':True,'profile':'FP32 policy plus FP32 frozen reference, 512-token serial LoRA step',
        'input_tokens':len(prompt)+len(response),'trainable_tensors':len(named),
        'trainable_parameters':sum(p.numel() for n,p in named),'changed_adapter_tensors':changed,
        'same_path_max_abs_logp_error':error,'sft_loss':loss.item(),'grad_norm_before_clip':norm.item(),
        'frozen_parameters_unchanged':True,'reference_gradients_absent':True,'probe_optimizer_steps':1,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    del policy,reference,optimizer,old,fresh,ref,loss,named,adapter_before
    gc.collect();torch.cuda.empty_cache()
    restored=PeftModel.from_pretrained(base(),adapter,is_trainable=True,autocast_adapter_dtype=False).eval()
    actual=response_scores(restored,prompt,response,False).cpu()
    reload_error=(expected-actual).abs().max().item()
    if reload_error>1e-5:raise ValueError('saved adapter reload differs')
    result['adapter_reload_max_abs_logp_error']=reload_error
    result['adapter_reload_passed']=True
    return result

def vllm_probe(output):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    import torch.distributed as dist
    cfg=read(ROOT/'configs/baseline.json');folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype=cfg['dtype'],trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=True,
        enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'])
    records=[]
    try:
        c=engine.llm_engine.get_vllm_config()
        effective={'dtype':str(c.model_config.dtype),'max_model_len':c.model_config.max_model_len,
            'logprobs_mode':c.model_config.logprobs_mode,'max_num_seqs':c.scheduler_config.max_num_seqs,
            'max_num_batched_tokens':c.scheduler_config.max_num_batched_tokens}
        if effective['max_model_len']!=cfg['max_model_len'] or effective['max_num_seqs']!=1:
            raise ValueError('effective engine profile differs')
        for label,width,max_tokens in [('short',128,64),('long_context',cfg['max_model_len']-16,16)]:
            prompt=make_prompt(tok,width)
            params=SamplingParams(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=max_tokens,
                logprobs=1,stop_token_ids=[tok.eos_token_id],detokenize=False)
            started=time.perf_counter()
            native=engine.generate([{'prompt_token_ids':prompt}],params,use_tqdm=False)[0]
            completion=native.outputs[0];ids=list(completion.token_ids)
            if list(native.prompt_token_ids)!=prompt or not ids or completion.logprobs is None:
                raise ValueError('native prompt/token records missing')
            if len(ids)!=len(completion.logprobs) or len(ids)>max_tokens:
                raise ValueError('native response alignment differs')
            raw=[entry[token].logprob for token,entry in zip(ids,completion.logprobs)]
            if any(not math.isfinite(v) or v>1e-6 for v in raw) or tok.eos_token_id in ids[:-1]:
                raise ValueError('invalid native logprobs or EOS')
            if completion.finish_reason not in ('stop','length'):
                raise ValueError('unexpected native finish reason')
            if (ids[-1]==tok.eos_token_id)!=(completion.finish_reason=='stop'):
                raise ValueError('native EOS and finish reason differ')
            if completion.finish_reason=='length' and len(ids)!=max_tokens:
                raise ValueError('native length stop differs')
            records.append({'probe':label,'prompt_tokens':len(prompt),'response_token_ids':ids,
                'raw_token_log_probs':raw,'text':tok.decode(ids,skip_special_tokens=True,
                    clean_up_tokenization_spaces=False),'finish_reason':completion.finish_reason,
                'seconds':time.perf_counter()-started})
        save(output/'vllm_records.json',records)
    finally:
        engine.llm_engine.engine_core.shutdown()
        if dist.is_initialized():dist.destroy_process_group()
    return {'passed':True,'effective_engine':effective,'probes':len(records),
        'max_prompt_tokens_tested':max(r['prompt_tokens'] for r in records),
        'training_admitted':False,'engine_shutdown_called':True}

def verdict(stages):
    passed=lambda key:stages.get(key,{}).get('passed') is True
    return {'cuda_ready':passed('inventory'),'model_cache_ready':passed('model'),
        'v2_baseline_ready':passed('inventory') and passed('model') and passed('vllm'),
        'short_context_lora_ready':passed('inventory') and passed('model') and passed('train'),
        'all_checks_passed':all(passed(k) for k in ('inventory','model','train','vllm')),
        'training_scope':'512-token FP32 serial LoRA with a resident frozen reference',
        'inference_scope':'V2 BF16 vLLM profile and near-8192-token context'}

def worker(args):
    started=time.perf_counter()
    try:
        if args.worker=='inventory':result=inventory(args.output)
        elif args.worker=='model':result=model_ready(args.output,args.download_model)
        elif args.worker=='train':result=train_probe(args.output)
        else:result=vllm_probe(args.output)
        result.update(elapsed_seconds=time.perf_counter()-started)
        save(args.output/('stage_'+args.worker+'.json'),result)
        return 0 if result.get('passed') else 2
    except Exception as exc:
        result={'passed':False,'error_type':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-started}
        save(args.output/('stage_'+args.worker+'.json'),result)
        traceback.print_exc()
        return 1

def run(args):
    if not math.isfinite(args.max_seconds) or args.max_seconds<=0:raise ValueError('positive time limit required')
    if args.hourly_price is not None and (not math.isfinite(args.hourly_price) or args.hourly_price<=0):
        raise ValueError('positive hourly price required')
    budget=min(args.max_seconds,900.)
    if args.hourly_price:budget=min(budget,10*3600/args.hourly_price)
    if budget<=5:raise ValueError('test budget is too short')
    tag=uuid.uuid4().hex[:8]
    output=ROOT/'artifacts'/('environment-4090d-20261030-'+tag)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),output/'check_environment.py')
    if (ROOT/'configs/baseline.json').exists():shutil.copyfile(ROOT/'configs/baseline.json',output/'baseline_config.json')
    started=time.perf_counter();stages={};error=None
    try:
        for stage in ('inventory','model','train','vllm'):
            if stage=='model' and not stages.get('inventory',{}).get('passed'):
                stages[stage]={'passed':False,'skipped':'CUDA or dependencies need attention'};continue
            if stage in ('train','vllm') and not stages.get('model',{}).get('passed'):
                stages[stage]={'passed':False,'skipped':'verified model cache required'};continue
            remaining=budget-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('environment check reached its time budget')
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',stage,'--output',str(output)]
            if args.download_model:command.append('--download-model')
            log_path=output/(stage+'.log')
            print('Checking '+stage+' ...',flush=True)
            with log_path.open('x',encoding='utf8') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                try:code=proc.wait(timeout=remaining)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                    else:proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                        else:proc.kill()
                        proc.wait()
                    raise TimeoutError('worker stopped at the diagnostic time budget')
            result_path=output/('stage_'+stage+'.json')
            stages[stage]=read(result_path) if result_path.exists() else {'passed':False,'exit_code':code,'log':log_path.name}
            print(stage+': '+('PASS' if stages[stage].get('passed') else 'NEEDS ATTENTION'),flush=True)
            # A train OOM does not prevent checking the independent vLLM inference profile.
    except (Exception,KeyboardInterrupt) as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        elapsed=time.perf_counter()-started
        summary={'diagnostic':'v2-4090d-environment-v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'verdict':verdict(stages),'stages':stages,'elapsed_seconds':elapsed,'error':error,
            'hourly_price_yuan':args.hourly_price,
            'estimated_job_cost_yuan':elapsed*args.hourly_price/3600 if args.hourly_price else None,
            'max_job_seconds':budget,'persistent_training_weights_modified':False,
            'probe_adapter_location':'only in this diagnostic result folder'}
        save(output/'summary.json',summary)
        files={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}
        save(output/'files_sha256.json',files)
        archive=ROOT/('v2-4090d-environment-review-20261030-'+tag+'.zip')
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(output.parent).as_posix())
        print(json.dumps(summary['verdict'],indent=2),flush=True)
        print('Download this file: '+str(archive),flush=True)
    return 0 if summary['verdict']['all_checks_passed'] else 2

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--download-model',action='store_true')
    parser.add_argument('--hourly-price',type=float)
    parser.add_argument('--max-seconds',type=float,default=900)
    parser.add_argument('--worker',choices=('inventory','model','train','vllm'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if args.worker:
        if args.output is None or not args.output.is_dir():raise ValueError('worker output directory required')
        return worker(args)
    return run(args)
if __name__=='__main__':raise SystemExit(main())
