"""Frozen-DPO sampling in train-only controlled recovery contexts, plus audited scripted negatives."""
import os,sys,json,math,shutil
from pathlib import Path
from dataclasses import asdict
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation,parse_action
from apt_v2.sql_sft import data_identity
from apt_v2.dpo import pair_id
from apt_v2.dpo_recovery import SCENARIOS,build_context,make_pair,encode_recovery_pair,select_pairs

def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs/dpo-recovery-collect-v1.json'):raise ValueError('fixed recovery config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    verify_files(ROOT,binding['source_files']);verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files'])
    verify_files(output,binding['prepared_files'])
    if read(output/'tasks.json')!=[asdict(t) for t in make_tasks() if t.split=='train']:raise ValueError('train tasks differ')
    return cfg,ROOT/'artifacts'/cfg['source_run']

def prepare(output):
    from transformers import AutoTokenizer
    cfg=read(output/'config.json');source=ROOT/'artifacts'/cfg['source_run'];parent=ROOT/'artifacts'/cfg['parent_run'];folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    if cfg['training_enabled'] is not False or cfg['split']!='train' or cfg['scenarios']!=list(SCENARIOS) or cfg['max_native_pairs_per_context']!=1:raise ValueError('fixed train-only recovery collection required')
    complete=read(source/'final/complete.json');verify_files(source/'final/adapter',complete['adapter_sha256'])
    if complete['algorithm']!='dpo' or complete['steps']!=11 or file_hash(source/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('reviewed frozen DPO required')
    dpo_cfg=read(source/'config.json')
    if dpo_cfg['adapter_sha256']!=cfg['sft_adapter_sha256'] or dpo_cfg['pairs_sha256']!=cfg['anchor_pairs_sha256']:raise ValueError('frozen policy provenance differs')
    identity=read(source/'model_identity.json')
    if identity!=read(parent/'model_identity.json') or identity['resolved_revision']!=cfg['revision']:raise ValueError('fixed base model differs')
    verify_files(folder,identity['files_sha256'])
    anchors_path=ROOT/'artifacts'/cfg['anchor_run']/'pairs.jsonl'
    if file_hash(anchors_path)!=cfg['anchor_pairs_sha256'] or file_hash(source/'pairs.jsonl')!=cfg['anchor_pairs_sha256']:raise ValueError('reviewed anchors differ')
    anchors=load(anchors_path);tasks=[t for t in make_tasks() if t.split=='train'];allowed={(t.id,t.family_id) for t in tasks}
    examples=load(parent/'examples.jsonl')
    if len(tasks)!=cfg['task_count'] or data_identity(examples)!=read(parent/'input_binding.json')['dataset_sha256']:raise ValueError('SFT training provenance differs')
    if len(anchors)!=44 or any(e['split']!='train' or (e['task_id'],e['family_id']) not in allowed for e in anchors+examples):raise ValueError('train-only provenance required')
    contexts=[build_context(t,s) for t in tasks for s in SCENARIOS]
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    scripted=[make_pair(tasks[[t.id for t in tasks].index(c['task_id'])],c,tok,cfg['pair_max_length']) for c in contexts]
    if any(p is None for p in scripted):raise ValueError('all scripted recovery scenarios must fit')
    for p in anchors:encode_recovery_pair(tok,p,cfg['pair_max_length'])
    write(output/'tasks.json',[asdict(t) for t in tasks]);write(output/'model_identity.json',identity)
    jsonl(output/'contexts.jsonl',contexts);jsonl(output/'scripted_pairs.jsonl',scripted);jsonl(output/'anchor_pairs.jsonl',anchors)
    paths=[source/'model_identity.json',source/'config.json',source/'pairs.jsonl',source/'final/complete.json',anchors_path,parent/'examples.jsonl',parent/'input_binding.json',parent/'model_identity.json']+[source/'final/adapter'/n for n in complete['adapter_sha256']]
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,p)
    prepared=['tasks.json','model_identity.json','contexts.jsonl','scripted_pairs.jsonl','anchor_pairs.jsonl']
    write(output/'input_binding.json',{'source_files':{p.relative_to(ROOT).as_posix():file_hash(p) for p in paths},
        'model_files':identity['files_sha256'],'prepared_files':{n:file_hash(output/n) for n in prepared}})
    write(output/'prepare_summary.json',{'train_tasks':len(tasks),'recovery_contexts':len(contexts),'scripted_pairs':len(scripted),
        'anchor_pairs':len(anchors),'planned_native_samples':len(contexts)*cfg['samples_per_context'],'training_enabled':False,'optimizer_updates':0})
    verify_inputs(output)

def collect(output):
    cfg,source=verify_inputs(output)
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    contexts=load(output/'contexts.jsonl');tasks=[t.id for t in make_tasks() if t.split=='train']
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,
        max_model_len=cfg['max_model_len'],max_num_seqs=1,max_num_batched_tokens=cfg['max_model_len'],gpu_memory_utilization=cfg['gpu_memory_utilization'],
        enforce_eager=True,enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=cfg['seed'],enable_lora=True,max_lora_rank=8,max_loras=2)
    request=LoRARequest('frozen-dpo-recovery',1,str(source/'final/adapter'));count=0
    try:
        with (output/'samples.jsonl').open('x',encoding='utf8') as stream:
            for context in contexts:
                prompt=tok.apply_chat_template(context['messages'],tokenize=True,add_generation_prompt=True)
                if len(prompt)+cfg['max_new_tokens']>cfg['max_model_len']:raise ValueError('context budget exceeded')
                for ri in range(cfg['samples_per_context']):
                    seed=cfg['seed']+tasks.index(context['task_id'])*10000+list(SCENARIOS).index(context['scenario'])*100+ri
                    params=SamplingParams(n=1,temperature=cfg['temperature'],top_p=1,top_k=-1,max_tokens=cfg['max_new_tokens'],logprobs=1,stop_token_ids=[tok.eos_token_id],detokenize=False,seed=seed)
                    native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                    if list(native.prompt_token_ids)!=prompt:raise ValueError('native prompt differs')
                    completion=native.outputs[0];ids=list(completion.token_ids)
                    if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native token alignment required')
                    raw=[p[t].logprob for t,p in zip(ids,completion.logprobs)]
                    if any(not math.isfinite(v) or v>1e-6 for v in raw):raise ValueError('finite raw log probabilities required')
                    item=asdict(Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),'dpo_sample',list(prompt),ids,raw,None,completion.finish_reason))
                    item.update(context_id=context['context_id'],task_id=context['task_id'],scenario=context['scenario'],sample_index=ri,
                        sample_id=context['context_id']+':sample-'+str(ri),sample_seed=seed,temperature=cfg['temperature'],adapter_sha256=cfg['adapter_sha256'])
                    stream.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();count+=1
                print(json.dumps({'context_id':context['context_id'],'samples_collected':count}),flush=True)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'collection_summary.json',{'native_samples':count,'samples_sha256':file_hash(output/'samples.jsonl'),'optimizer_updates':0})
    verify_inputs(output)

def derive(output):
    from transformers import AutoTokenizer
    cfg,_=verify_inputs(output);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    for n,h in read(output/'model_identity.json')['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tasks=[t for t in make_tasks() if t.split=='train'];task_map={t.id:t for t in tasks}
    contexts=load(output/'contexts.jsonl');expected=[build_context(t,s) for t in tasks for s in SCENARIOS]
    if contexts!=expected:raise ValueError('recovery context manifest differs')
    context_map={c['context_id']:c for c in contexts};scripted=[make_pair(task_map[c['task_id']],c,tok,cfg['pair_max_length']) for c in contexts]
    if scripted!=load(output/'scripted_pairs.jsonl'):raise ValueError('scripted preference replay differs')
    samples=load(output/'samples.jsonl');wanted={(c['context_id'],i) for c in contexts for i in range(cfg['samples_per_context'])}
    if len(samples)!=len(wanted) or {(s['context_id'],s['sample_index']) for s in samples}!=wanted:raise ValueError('complete distinct recovery samples required')
    collection=read(output/'collection_summary.json')
    if collection['native_samples']!=len(samples) or collection['samples_sha256']!=file_hash(output/'samples.jsonl'):raise ValueError('native sample summary differs')
    native=[]
    for sample in samples:
        c=context_map[sample['context_id']];ti=list(task_map).index(c['task_id']);si=list(SCENARIOS).index(c['scenario']);ri=sample['sample_index']
        if sample['task_id']!=c['task_id'] or sample['scenario']!=c['scenario'] or sample['sample_seed']!=cfg['seed']+ti*10000+si*100+ri or sample['temperature']!=cfg['temperature'] or sample['adapter_sha256']!=cfg['adapter_sha256'] or sample['sample_id']!=c['context_id']+':sample-'+str(ri):raise ValueError('native sample identity differs')
        pair=make_pair(task_map[c['task_id']],c,tok,cfg['pair_max_length'],sample)
        if pair is not None:native.append(pair)
    anchors=load(output/'anchor_pairs.jsonl')
    if len(anchors)!=44 or any(p['pair_id']!=pair_id(p) for p in anchors):raise ValueError('anchor identity differs')
    pairs=select_pairs(anchors,scripted,native)
    for p in pairs:encode_recovery_pair(tok,p,cfg['pair_max_length'])
    summary={'pairs':len(pairs),'preserved_sql_anchors':44,'scripted_recovery_pairs':len(scripted),
        'native_failure_candidates':len(native),'selected_native_recovery_pairs':sum(p['rejected_source']=='frozen_dpo_sample' for p in pairs),
        'native_samples_audited':len(samples),'recovery_contexts':len(contexts),'task_coverage':len({p['task_id'] for p in pairs}),
        'pairs_by_chosen_tool':dict(Counter(parse_action(p['chosen'])['tool'] for p in pairs)),
        'pairs_by_rejected_source':dict(Counter(p['rejected_source'] for p in pairs)),
        'recovery_pairs_by_scenario':dict(Counter(p['scenario'] for p in pairs if 'scenario' in p)),
        'ready_for_training_review':len(scripted)==84 and len({p['task_id'] for p in pairs})==12,
        'training_enabled':False,'optimizer_updates':0}
    return pairs,native,summary

def build(output):
    pairs,native,summary=derive(output)
    jsonl(output/'native_pairs.jsonl',native);jsonl(output/'pairs.jsonl',pairs)
    write(output/'pair_summary.json',{**summary,'pairs_sha256':file_hash(output/'pairs.jsonl')})

def audit(output):
    pairs,native,summary=derive(output)
    if pairs!=load(output/'pairs.jsonl') or native!=load(output/'native_pairs.jsonl') or {**summary,'pairs_sha256':file_hash(output/'pairs.jsonl')}!=read(output/'pair_summary.json'):raise ValueError('recovery dataset replay differs')
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,'contexts_replayed':84,'native_samples_audited':summary['native_samples_audited'],
        'pairs_verified':len(pairs),'scripted_vs_native_sources_separated':True,'train_only':True,'baseline_unchanged':True,'optimizer_updates':0})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','collect','build','audit'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1');globals()[a.stage](a.output)
