"""Inference-only paired test of parent and weighted SFT on independent task families."""
import sys,os,json,math,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.runtime import Generation,run_episode
from apt_v2.tasks import make_tasks
from apt_v2.chosen_sft import chosen_examples,data_sha256
from apt_v2.recovery_diagnostics import messages_id,summarize
from apt_v2.independent_recovery_test import independent_tasks,test_contexts,diagnose_test_action,run_test_recovery,TEST_SCENARIOS,NOVEL_SCENARIOS
from apt_v2.sft_diagnostics import summarize_episodes

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def checkpoints(cfg):return {'before':ROOT/'artifacts'/cfg['parent_run']/'final','after':ROOT/'artifacts'/cfg['candidate_run']/'final'}
def recovery_seed(cfg,tasks,c,ri):return cfg['recovery_eval']['seed']+[t.id for t in tasks].index(c['task_id'])*100000+list(TEST_SCENARIOS).index(c['scenario'])*1000+ri*100

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs/hard-recovery-independent-test-v1.json') or cfg['training_enabled'] is not False:raise ValueError('fixed inference-only independent test required')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,binding['provenance'])
    for branch,p in checkpoints(cfg).items():
        verify_files(p,binding['checkpoints'][branch])
        expected=cfg['adapter_sha256'] if branch=='before' else cfg['candidate_adapter_sha256']
        if file_hash(p/'adapter/adapter_model.safetensors')!=expected:raise ValueError('pinned test adapter differs')
    if binding['training_enabled'] is not False or binding['optimizer_updates']!=0:raise ValueError('inference-only binding required')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files'])
    for name,key in [('pairs.jsonl','pairs_sha256'),('examples.jsonl','examples_file_sha256'),('test_tasks.json','test_tasks_sha256'),('contexts.jsonl','contexts_sha256')]:
        if file_hash(output/name)!=binding[key]:raise ValueError('independent test input differs: '+name)
    if file_hash(output/'config.json')!=binding['config_sha256']:raise ValueError('test config binding differs')
    if read(output/'model_identity.json')['files_sha256']!=binding['model_files']:raise ValueError('saved base identity differs')
    pairs=load(output/'pairs.jsonl');examples=chosen_examples(pairs);tasks=independent_tasks(pairs,examples)
    if examples!=load(output/'examples.jsonl') or read(output/'test_tasks.json')!=[asdict(t) for t in tasks]:raise ValueError('test task or training projection differs')
    return cfg

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs/hard-recovery-independent-test-v1.json') or cfg['training_enabled'] is not False:raise ValueError('fixed independent evaluation required')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    parent=ROOT/'artifacts'/cfg['parent_run'];candidate=ROOT/'artifacts'/cfg['candidate_run'];data=ROOT/'artifacts'/cfg['data_run']
    identities=[];bound={}
    for branch,run in [('before',parent),('after',candidate)]:
        complete=read(run/'final/complete.json');verify_files(run/'final/adapter',complete['adapter_sha256'])
        expected=cfg['adapter_sha256'] if branch=='before' else cfg['candidate_adapter_sha256']
        if file_hash(run/'final/adapter/adapter_model.safetensors')!=expected or not read(run/'audit.json')['passed']:raise ValueError('reviewed pinned checkpoint required')
        bound[branch]={'complete.json':file_hash(run/'final/complete.json'),**{'adapter/'+n:h for n,h in complete['adapter_sha256'].items()}}
        identities.append(read(run/'model_identity.json'))
    if identities[0]!=identities[1] or identities[0]['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('fixed base model differs')
    pc=read(parent/'final/complete.json');cc=read(candidate/'final/complete.json');tc=read(candidate/'config.json');ts=read(candidate/'training_summary.json')
    if pc['total_sft_updates']!=255 or pc['total_dpo_updates']!=11 or cc['total_sft_updates']!=513 or cc['total_dpo_updates']!=11 or cc['steps']!=258 or ts['optimizer_updates']!=258:
        raise ValueError('reviewed SFT ancestry differs')
    if tc['initial_run']!=cfg['parent_run'] or tc['adapter_sha256']!=cfg['adapter_sha256'] or tc['pairs_sha256']!=cfg['pairs_sha256']:raise ValueError('weighted training provenance differs')
    if not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed train pairs required')
    pairs=load(data/'pairs.jsonl');examples=chosen_examples(pairs);tasks=independent_tasks(pairs,examples)
    if len(tasks)!=cfg['sample_eval']['task_count'] or len(tasks)!=cfg['recovery_eval']['task_count']:raise ValueError('fixed independent task count required')
    for run in (parent,candidate):
        if file_hash(run/'pairs.jsonl')!=cfg['pairs_sha256'] or load(run/'examples.jsonl')!=examples:raise ValueError('model train datasets differ')
    contexts=test_contexts(tasks);seen={messages_id(p['messages']) for p in pairs}
    if any(messages_id(c['messages']) in seen for c in contexts):raise ValueError('test recovery context overlaps training')
    identity=identities[0];folder=ROOT/'.cache/qwen2.5-1.5b-instruct';verify_files(folder,identity['files_sha256'])
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');jsonl(output/'examples.jsonl',examples)
    write(output/'test_tasks.json',[asdict(t) for t in tasks]);jsonl(output/'contexts.jsonl',contexts);write(output/'model_identity.json',identity)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,p)
    provenance=[data/'pairs.jsonl',data/'audit.json']
    for run in (parent,candidate):
        provenance.extend(run/n for n in ('config.json','audit.json','training_summary.json','input_binding.json','pairs.jsonl','examples.jsonl','model_identity.json'))
    write(output/'input_binding.json',{'checkpoints':bound,'model_files':identity['files_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance},'pairs_sha256':cfg['pairs_sha256'],
        'examples_file_sha256':file_hash(output/'examples.jsonl'),'test_tasks_sha256':file_hash(output/'test_tasks.json'),
        'contexts_sha256':file_hash(output/'contexts.jsonl'),'config_sha256':file_hash(output/'config.json'),'training_enabled':False,'optimizer_updates':0})
    write(output/'prepare_summary.json',{'test_tasks':4,'fault_tasks':2,'task_families':[t.family_id for t in tasks],
        'greedy_episodes':8,'sample_episodes':64,'standard_recovery_contexts':28,'novel_recovery_contexts':8,
        'first_recovery_actions':288,'novel_recovery_continuations':64,'test_contexts_in_training':0,'training_enabled':False,'optimizer_updates':0})
    verify_inputs(output)

def engine_and_tokenizer(output,cfg):
    from transformers import AutoTokenizer
    from vllm import LLM
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';ev=cfg['eval'];tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],
        max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,
        enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=ev['seed'],enable_lora=True,max_lora_rank=8,max_loras=2)
    return engine,tok

def shutdown(engine):
    engine.llm_engine.engine_core.shutdown()
    import torch.distributed as dist
    if dist.is_initialized():dist.destroy_process_group()

class NativePolicy:
    def __init__(self,engine,tokenizer,lora,cfg,mode,seed,temperature):
        self.engine=engine;self.tok=tokenizer;self.lora=lora;self.cfg=cfg;self.mode=mode;self.seed=seed;self.temperature=temperature;self.calls=0
    def __call__(self,messages):
        from vllm import SamplingParams
        prompt=self.tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True);ev=self.cfg['eval']
        if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('test prompt budget exceeded')
        params=SamplingParams(n=1,temperature=self.temperature,top_p=1,top_k=-1,max_tokens=ev['max_new_tokens'],logprobs=1,
            stop_token_ids=[self.tok.eos_token_id],detokenize=False,seed=self.seed+self.calls);self.calls+=1
        native=self.engine.generate([{'prompt_token_ids':prompt}],params,lora_request=self.lora,use_tqdm=False)[0]
        if list(native.prompt_token_ids)!=list(prompt):raise ValueError('native test prompt differs')
        c=native.outputs[0];ids=list(c.token_ids)
        if not ids or c.logprobs is None or len(ids)!=len(c.logprobs):raise ValueError('native test token alignment required')
        raw=[p[t].logprob for t,p in zip(ids,c.logprobs)]
        if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite native test probabilities required')
        return Generation(self.tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
            'greedy' if self.mode=='greedy' else 'dpo_sample',list(prompt),ids,raw,None,c.finish_reason)

def complete_tasks(output,mode):
    cfg=verify_inputs(output);tasks=independent_tasks(load(output/'pairs.jsonl'),load(output/'examples.jsonl'))
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    engine,tok=engine_and_tokenizer(output,cfg);sc=cfg['sample_eval'];summaries={}
    try:
        total=len(tasks)*len(checkpoints(cfg))*(1 if mode=='greedy' else sc['rollouts_per_task'])
        with tqdm(total=total,desc='Independent '+mode,unit='episode',file=sys.stdout,disable=False,dynamic_ncols=True) as progress:
            for bi,(branch,p) in enumerate(checkpoints(cfg).items(),1):
                lora=LoRARequest(branch,bi,str(p/'adapter'));episodes=[];name=branch+('-test' if mode=='greedy' else '-sample')
                with (output/(name+'-episodes.jsonl')).open('x',encoding='utf8') as stream:
                    for ti,task in enumerate(tasks):
                        for ri in range(1 if mode=='greedy' else sc['rollouts_per_task']):
                            seed=(cfg['eval']['seed']+ti*10000) if mode=='greedy' else sc['seed']+ti*10000+ri*100
                            policy=NativePolicy(engine,tok,lora,cfg,mode,seed,0 if mode=='greedy' else sc['temperature'])
                            try:e=run_episode(task,policy)
                            except Exception as exc:
                                if hasattr(exc,'episode_evidence'):write(output/('partial-'+name+'.json'),exc.episode_evidence)
                                raise
                            e['model_branch']=branch
                            if mode=='sample':
                                e.update(rollout_index=ri,sample_seed=seed,temperature=sc['temperature'],rollout_id=f'{task.id}-test-{branch}-{ri:02}')
                                for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                            episodes.append(e);stream.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                            progress.set_postfix(branch=branch,task=task.id,success=e['success'],refresh=False);progress.update(1)
                summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(output/(name+'-episodes.jsonl'))}
                if mode=='sample':summary['successes_by_task']={t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in tasks}
                write(output/(name+'-summary.json'),summary);summaries[branch]={'test':summary} if mode=='greedy' else summary
    finally:shutdown(engine)
    write(output/('comparison_summary.json' if mode=='greedy' else 'stochastic_comparison.json'),summaries);verify_inputs(output)

def recovery_summary(actions,episodes,branches):
    result={}
    for branch in branches:
        result[branch]={}
        for mode in ('greedy','sample'):
            rows=[r for r in actions if r['model_branch']==branch and r['mode']==mode]
            es=[e for e in episodes if e['model_branch']==branch and e['mode']==mode]
            def esummary(group):return {'episodes':len(group),'successes':sum(e['success'] for e in group),'turns':sum(len(e['turns']) for e in group),'tool_errors':sum(e['tool_errors'] for e in group)}
            result[branch][mode]={'first_action':{'overall':summarize(rows),'by_scenario':{s:summarize([r for r in rows if r['scenario']==s]) for s in TEST_SCENARIOS}},
                'continuation':{'overall':esummary(es),'by_scenario':{s:esummary([e for e in es if e['scenario']==s]) for s in NOVEL_SCENARIOS}}}
    return result

def recovery(output):
    cfg=verify_inputs(output);tasks=independent_tasks(load(output/'pairs.jsonl'),load(output/'examples.jsonl'));tm={t.id:t for t in tasks};contexts=load(output/'contexts.jsonl')
    if contexts!=test_contexts(tasks):raise ValueError('fixed test recovery contexts differ')
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    engine,tok=engine_and_tokenizer(output,cfg);rc=cfg['recovery_eval'];actions=[];episodes=[]
    try:
        total=len(contexts)*(1+rc['samples_per_context'])*len(checkpoints(cfg))
        with tqdm(total=total,desc='Independent recovery',unit='context-run',file=sys.stdout,disable=False,dynamic_ncols=True) as progress:
            for bi,(branch,p) in enumerate(checkpoints(cfg).items(),1):
                lora=LoRARequest(branch,bi,str(p/'adapter'))
                with (output/(branch+'-recovery-actions.jsonl')).open('x',encoding='utf8') as af,(output/(branch+'-recovery-episodes.jsonl')).open('x',encoding='utf8') as ef:
                    for c in contexts:
                        task=tm[c['task_id']]
                        for mode in ('greedy','sample'):
                            for ri in range(1 if mode=='greedy' else rc['samples_per_context']):
                                seed=recovery_seed(cfg,tasks,c,ri);temperature=0 if mode=='greedy' else rc['temperature']
                                policy=NativePolicy(engine,tok,lora,cfg,mode,seed,temperature)
                                if c['scenario'] in NOVEL_SCENARIOS:
                                    try:e=run_test_recovery(task,c,policy)
                                    except Exception as exc:
                                        if hasattr(exc,'episode_evidence'):write(output/('partial-recovery-'+branch+'.json'),exc.episode_evidence)
                                        raise
                                    e.update(model_branch=branch,mode=mode,sample_index=ri,sample_seed=seed,temperature=temperature)
                                    for i,row in enumerate(e['turns']):row['sampling_seed']=seed+i
                                    g=e['turns'][0]['generation'];episodes.append(e);ef.write(json.dumps(e,ensure_ascii=False,allow_nan=False)+'\n');ef.flush()
                                else:g=asdict(policy(c['messages']))
                                actual,d=diagnose_test_action(task,c,g['text'])
                                row={'model_branch':branch,'task_id':task.id,'family_id':task.family_id,'split':'test','context_id':c['context_id'],
                                    'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d}
                                actions.append(row);af.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');af.flush()
                                progress.set_postfix(branch=branch,scenario=c['scenario'],refresh=False);progress.update(1)
    finally:shutdown(engine)
    write(output/'recovery_comparison.json',recovery_summary(actions,episodes,checkpoints(cfg)));verify_inputs(output)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=['prepare','greedy','sample','recovery'],required=True)
    a=p.parse_args()
    if a.stage in ('greedy','sample'):complete_tasks(a.output,a.stage)
    else:globals()[a.stage](a.output)
