"""Finite inference-only SFT diagnosis; each backend runs in a fresh process."""
import sys,os,json,math,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from common import read,write,file_hash,verify_files
from apt_v2.sft import build_dataset,encode_example
from apt_v2.runtime import Generation,parse_action,run_episode
from apt_v2.tasks import make_tasks
from apt_v2.sft_diagnostics import select_probes,score_action,grouped_scores,summarize_episodes,compare_scores,probe_record,index_probes

def emit(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def context(output):
    cfg=read(output/'config.json');pilot=ROOT/'artifacts'/cfg['pilot_run'];model=ROOT/'.cache/qwen2.5-1.5b-instruct'
    binding=read(output/'input_binding.json')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    verify_files(pilot,binding['pilot_files']);verify_files(model,binding['model_files'])
    return cfg,pilot,model

def prepare(output):
    cfg=read(output/'config.json');pilot=ROOT/'artifacts'/cfg['pilot_run'];model=ROOT/'.cache/qwen2.5-1.5b-instruct'
    identity=read(pilot/'model_identity.json');verify_files(model,identity['files_sha256'])
    if identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('model revision differs')
    old=read(pilot/'run_manifest.json')['source_sha256'];verify_files(ROOT,old)
    complete=read(pilot/'final/complete.json');verify_files(pilot/'final/adapter',complete['adapter_sha256'])
    names=['examples.jsonl','data_summary.json','config.json','run_manifest.json','model_identity.json',
        'final/complete.json','final/binding.json']+['final/adapter/'+n for n in complete['adapter_sha256']]
    binding={'pilot_files':{n:file_hash(pilot/n) for n in names},'model_files':identity['files_sha256']}
    write(output/'input_binding.json',binding)
    examples=[json.loads(line) for line in (pilot/'examples.jsonl').read_text(encoding='utf8').splitlines()]
    _,rebuilt=build_dataset()
    if examples!=rebuilt:raise ValueError('training examples differ from replay-verified generator')
    if file_hash(pilot/'examples.jsonl')!=read(pilot/'data_summary.json')['examples_sha256']:raise ValueError('dataset identity differs')
    probes=select_probes(examples)
    if len(probes)!=cfg['probe_count']:raise ValueError('probe count differs')
    write(output/'probes.json',probes)
    # Freeze small input evidence in the review, without duplicating optimizer checkpoints.
    write(output/'pilot_identity.json',{'model':identity,'final':complete,'binding':read(pilot/'final/binding.json')})
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False)
    enc=[encode_example(tok,e,cfg['max_length']) for e in examples]
    for probe in probes:
        encoded=encode_example(tok,probe,cfg['max_length'])
        probe['prompt_token_ids']=encoded['input_ids'][:encoded['prompt_length']]
        probe['response_token_ids']=encoded['input_ids'][encoded['prompt_length']:]
    write(output/'encoded_probes.json',probes)
    for name in identity['files_sha256']:
        if name.endswith(('.json','.txt','.jinja')):
            import shutil
            target=output/'tokenizer'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(model/name,target)
    write(output/'prepare_summary.json',{'examples':len(examples),'max_tokens':max(len(s['input_ids']) for s in enc),
        'probes':len(probes),'training_enabled':False,'optimizer_updates':0})

def hf(output,precision):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
    from peft import PeftModel
    cfg,pilot,folder=context(output)
    if not torch.cuda.is_available():raise ValueError('CUDA required')
    torch.manual_seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    dtype=torch.float32 if precision=='fp32' else torch.bfloat16
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
        torch_dtype=dtype,attn_implementation='sdpa').cuda().eval()
    model=PeftModel.from_pretrained(base,pilot/'final/adapter',is_trainable=False,autocast_adapter_dtype=False).eval()
    model.requires_grad_(False)
    probes=read(output/'encoded_probes.json');rows=[];probe_rows=[]
    if precision=='fp32':
        examples=[json.loads(line) for line in (pilot/'examples.jsonl').read_text(encoding='utf8').splitlines()]
        with (output/'hf-fp32-actions.jsonl').open('x',encoding='utf8') as stream:
            for mode in ('base','sft'):
                with model.disable_adapter() if mode=='base' else contextlib.nullcontext():
                    for i,example in enumerate(examples):
                        sample=encode_example(tok,example,cfg['max_length']);score=score_action(model,sample)
                        row={k:score[k] for k in ('tokens','nll_sum','correct_tokens','all_tokens_correct')}
                        row.update(mode=mode,example_index=i,tool=parse_action(example['response'])['tool'],
                            variant=example['variant'],task_id=example['task_id'])
                        rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps({'stage':'teacher-forced','mode':mode,'examples':len(examples)}),flush=True)
        write(output/'hf-fp32-score-summary.json',{m:grouped_scores([r for r in rows if r['mode']==m]) for m in ('base','sft')})
    for mode in ('base','sft'):
        with model.disable_adapter() if mode=='base' else contextlib.nullcontext():
            for probe in probes:
                sample=encode_example(tok,probe,cfg['max_length']);score=score_action(model,sample)
                ids=torch.tensor([probe['prompt_token_ids']],device='cuda')
                with torch.inference_mode():
                    full=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
                        generation_config=GenerationConfig(do_sample=False,max_new_tokens=cfg['max_new_tokens'],
                            temperature=1.,top_p=1.,top_k=0,repetition_penalty=1.,num_beams=1,
                            eos_token_id=tok.eos_token_id,pad_token_id=tok.eos_token_id,use_cache=True))
                response=full[0,len(probe['prompt_token_ids']):].tolist()
                probe_rows.append({'probe_id':probe['probe_id'],'mode':mode,'precision':precision,
                    'score':score,'prompt_token_ids':probe['prompt_token_ids'],'response_token_ids':response,
                    'text':tok.decode(response,skip_special_tokens=True,clean_up_tokenization_spaces=False)})
    write(output/('hf-'+precision+'-probes.json'),probe_rows)
    write(output/('hf-'+precision+'-status.json'),{'training_enabled':False,'optimizer_updates':0,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'adapter_dtypes':sorted({str(p.dtype) for n,p in model.named_parameters() if 'lora_' in n}),
        'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'dtype':str(dtype)})

def vllm(output):
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    cfg,pilot,folder=context(output)
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],
        gpu_memory_utilization=cfg['gpu_memory_utilization'],enforce_eager=True,enable_prefix_caching=False,
        generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'],enable_lora=True,max_lora_rank=8)
    request=LoRARequest('sft-pilot-diagnostics',1,str(pilot/'final/adapter'))
    params=SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=cfg['max_new_tokens'],
        stop_token_ids=[tok.eos_token_id],detokenize=False,logprobs=1)
    teacher=SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=1,
        stop_token_ids=[tok.eos_token_id],detokenize=False,prompt_logprobs=1)
    probes=read(output/'encoded_probes.json');probe_rows=[];summaries={}
    def generate(prompt,mode):
        if len(prompt)+cfg['max_new_tokens']>cfg['max_model_len']:raise ValueError('context budget exceeded')
        native=engine.generate([{'prompt_token_ids':prompt}],params,
            lora_request=request if mode=='sft' else None,use_tqdm=False)[0]
        if list(native.prompt_token_ids)!=prompt:raise ValueError('prompt identity differs')
        completion=native.outputs[0];ids=list(completion.token_ids)
        if not ids or completion.logprobs is None or len(completion.logprobs)!=len(ids):raise ValueError('native probability alignment')
        raw=[r[t].logprob for t,r in zip(ids,completion.logprobs)]
        if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite log probabilities required')
        return Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            'greedy',prompt,ids,raw,None,completion.finish_reason)
    try:
        for mode in ('base','sft'):
            for probe in probes:
                sample=generate(probe['prompt_token_ids'],mode)
                prompt=probe['prompt_token_ids'];gold=probe['response_token_ids'];full=prompt+gold
                native=engine.generate([{'prompt_token_ids':full}],teacher,
                    lora_request=request if mode=='sft' else None,use_tqdm=False)[0]
                if list(native.prompt_token_ids)!=full:raise ValueError('teacher-forced prompt identity differs')
                all_rows=native.prompt_logprobs
                if all_rows is None or len(all_rows)!=len(full):raise ValueError('prompt logprob alignment')
                selected=all_rows[len(prompt):]
                logp=[r[t].logprob for t,r in zip(gold,selected)]
                top=[max(r,key=lambda t:r[t].logprob) for r in selected]
                if any(not math.isfinite(v) for v in logp):raise ValueError('finite teacher-forced probabilities required')
                score={'token_log_probs':logp,'argmax_token_ids':top,'tokens':len(gold),
                    'nll_sum':-sum(logp),'correct_tokens':sum(a==b for a,b in zip(top,gold)),
                    'all_tokens_correct':top==gold}
                probe_rows.append(probe_record(probe['probe_id'],mode,sample,score))
            episodes=[]
            with (output/('train-'+mode+'-episodes.jsonl')).open('x',encoding='utf8') as stream:
                tasks=[t for t in make_tasks() if t.split=='train']
                if len(tasks)!=cfg['train_task_count']:raise ValueError('train task count differs')
                for task in tasks:
                    def policy(messages):
                        return generate(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True),mode)
                    try:episode=run_episode(task,policy)
                    except Exception as exc:
                        if hasattr(exc,'episode_evidence'):write(output/('partial-'+mode+'.json'),exc.episode_evidence)
                        raise
                    episodes.append(episode);stream.write(json.dumps(episode,ensure_ascii=False)+'\n');stream.flush()
                    print(json.dumps({'mode':mode,'task':task.id,'success':episode['success']}),flush=True)
            summaries[mode]=summarize_episodes(episodes)
            write(output/('train-'+mode+'-summary.json'),summaries[mode])
            write(output/('vllm-'+mode+'-probes.json'),[r for r in probe_rows if r['mode']==mode])
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'vllm-probes.json',probe_rows)

def audit(output):
    cfg,pilot,folder=context(output)
    from apt_v2.environment import Environment
    tasks={t.id:t for t in make_tasks()};replayed=0
    for mode in ('base','sft'):
        path=output/('train-'+mode+'-episodes.jsonl')
        episodes=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        if len(episodes)!=cfg['train_task_count'] or {e['task_id'] for e in episodes}!={t.id for t in tasks.values() if t.split=='train'}:
            raise ValueError('complete distinct train task set required')
        for episode in episodes:
            if episode['split']!='train':raise ValueError('train-only diagnostic episodes required')
            env=Environment(tasks[episode['task_id']])
            messages=[{'role':'system','content':__import__('apt_v2.runtime',fromlist=['SYSTEM']).SYSTEM},
                {'role':'user','content':tasks[episode['task_id']].request()}]
            from transformers import AutoTokenizer
            tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True)
            try:
                for row in episode['turns']:
                    g=row['generation']
                    if g['prompt_token_ids']!=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True):raise ValueError('replay prompt differs')
                    if tok.decode(g['response_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=g['text']:raise ValueError('replay decode differs')
                    try:action=parse_action(g['text'])
                    except (TypeError,ValueError):action={'invalid_action':g['text']}
                    actual=env.step(action)
                    for key in ('action','observation','reward','done','success','tool_error','new_verified_metrics'):
                        if actual[key]!=row[key]:raise ValueError('replay mismatch: '+key)
                    messages += [{'role':'assistant','content':g['text']},{'role':'user','content':json.dumps({'observation':row['observation']})}]
                if not env.done or env.success!=episode['success']:raise ValueError('replay completion differs')
            finally:env.close()
            replayed+=1
        if summarize_episodes(episodes)!=read(output/('train-'+mode+'-summary.json')):raise ValueError('summary differs')
    probe_ids=[p['probe_id'] for p in read(output/'encoded_probes.json')]
    backends={name:index_probes(read(output/file),probe_ids) for name,file in
        [('hf_fp32','hf-fp32-probes.json'),('hf_bf16','hf-bf16-probes.json'),('vllm','vllm-probes.json')]}
    comparisons=[]
    for left,right in [('hf_fp32','hf_bf16'),('hf_bf16','vllm')]:
        if set(backends[left])!=set(backends[right]):raise ValueError('probe set differs')
        for key,a in backends[left].items():
            b=backends[right][key]
            if a['prompt_token_ids']!=b['prompt_token_ids']:raise ValueError('cross-engine prompt differs')
            comparisons.append({'left':left,'right':right,'mode':key[0],'probe_id':key[1],
                'greedy_tokens_equal':a['response_token_ids']==b['response_token_ids'],
                **compare_scores(a['score'],b['score'])})
    effects={}
    for name,rows in backends.items():
        effects[name]=[{'probe_id':p['probe_id'],
            'greedy_tokens_changed':rows['base',p['probe_id']]['response_token_ids']!=rows['sft',p['probe_id']]['response_token_ids'],
            **compare_scores(rows['base',p['probe_id']]['score'],rows['sft',p['probe_id']]['score'])}
            for p in read(output/'encoded_probes.json')]
    write(output/'comparison.json',{'comparisons':comparisons,'adapter_effects':effects,'replayed_episodes':replayed,
        'training_enabled':False,'optimizer_updates':0})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','hf-fp32','hf-bf16','vllm','audit'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    if a.stage=='prepare':prepare(a.output)
    elif a.stage.startswith('hf-'):hf(a.output,a.stage.split('-')[1])
    elif a.stage=='vllm':vllm(a.output)
    else:audit(a.output)
