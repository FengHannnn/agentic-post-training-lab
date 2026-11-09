"""Generate and audit same-setting recovery probes for training and development contexts."""
import sys,os,json,math,shutil,csv
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.runtime import Generation
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.dpo_recovery_training import validate_training_pairs,validate_generation
from apt_v2.recovery_diagnostics import diagnostic_contexts,diagnose_action,diagnostic_summary,messages_id,comparison_rates

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def selected_tasks(cfg):
    tasks=[t for t in make_tasks() if t.split in cfg['task_counts']]
    for split,count in cfg['task_counts'].items():
        if split not in ('train','development') or sum(t.split==split for t in tasks)!=count:raise ValueError('fixed train/development tasks required')
    return tasks
def sampling_seed(cfg,tasks,context,index):
    split=context['split'];subset=[t for t in tasks if t.split==split];rc=cfg['recovery_eval']
    return rc['seed']+rc['split_seed_offsets'][split]+[t.id for t in subset].index(context['task_id'])*10000+list(SCENARIOS).index(context['scenario'])*100+index

def verify_inputs(output,prepared=True):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs/recovery-context-diagnostics-v1.json') or cfg['algorithm']!='inference_only_recovery_diagnostic':
        raise ValueError('fixed inference-only diagnostic config required')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';identity=None
    for branch,spec in cfg['checkpoints'].items():
        run=ROOT/'artifacts'/spec['run'];complete=read(run/'final/complete.json')
        verify_files(run/'final/adapter',complete['adapter_sha256'])
        if file_hash(run/'final/adapter/adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('pinned adapter differs: '+branch)
        current=read(run/'model_identity.json')
        if identity is None:identity=current
        elif current!=identity:raise ValueError('shared base model differs')
        if not read(run/'audit.json')['passed']:raise ValueError('audited checkpoint required')
    data=ROOT/'artifacts'/cfg['data_run']
    if file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256'] or not read(data/'audit.json')['passed']:raise ValueError('reviewed train pairs required')
    verify_files(folder,identity['files_sha256'])
    if prepared:
        binding=read(output/'input_binding.json');verify_files(ROOT,binding['provenance'])
        if binding['config_sha256']!=file_hash(output/'config.json') or binding['pairs_sha256']!=file_hash(output/'pairs.jsonl'):raise ValueError('diagnostic binding differs')
        if read(output/'model_identity.json')!=identity:raise ValueError('saved model identity differs')
    return cfg,identity

def prepare(output):
    cfg,identity=verify_inputs(output,prepared=False)
    data=ROOT/'artifacts'/cfg['data_run'];pairs=load(data/'pairs.jsonl');validate_training_pairs(pairs)
    tasks=selected_tasks(cfg);contexts=diagnostic_contexts(tasks)
    train_ids={messages_id(p['messages']) for p in pairs}
    membership=[{'context_id':c['context_id'],'task_id':c['task_id'],'split':c['split'],'messages_sha256':messages_id(c['messages']),
        'seen_in_training_pairs':messages_id(c['messages']) in train_ids} for c in contexts]
    if any(m['seen_in_training_pairs']!=(m['split']=='train') for m in membership):raise ValueError('exact training context membership differs')
    jsonl(output/'contexts.jsonl',contexts);jsonl(output/'context_membership.jsonl',membership)
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');write(output/'model_identity.json',identity)
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            target=output/'tokenizer'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,target)
    provenance=[data/'audit.json',data/'pairs.jsonl']
    for spec in cfg['checkpoints'].values():
        run=ROOT/'artifacts'/spec['run']
        provenance.extend([run/'model_identity.json',run/'audit.json',run/'final/complete.json'])
        provenance.extend(run/'final/adapter'/n for n in read(run/'final/complete.json')['adapter_sha256'])
    write(output/'input_binding.json',{'config_sha256':file_hash(output/'config.json'),'pairs_sha256':cfg['pairs_sha256'],
        'contexts_sha256':file_hash(output/'contexts.jsonl'),'membership_sha256':file_hash(output/'context_membership.jsonl'),
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance}})
    write(output/'prepare_summary.json',{'contexts':len(contexts),'train_contexts':84,'development_contexts':28,'training_contexts_exactly_in_pairs':84,
        'development_contexts_in_training_pairs':0,'planned_actions':len(contexts)*4*3,'training_updates':0})
    verify_inputs(output)

def evaluate(output):
    cfg,_=verify_inputs(output);tasks=selected_tasks(cfg);tm={t.id:t for t in tasks};contexts=load(output/'contexts.jsonl')
    if contexts!=diagnostic_contexts(tasks):raise ValueError('fixed contexts differ')
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from tqdm.auto import tqdm
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    ev=cfg['eval'];rc=cfg['recovery_eval']
    engine=LLM(model=str(folder),tokenizer=str(folder),dtype='bfloat16',trust_remote_code=False,max_model_len=ev['max_model_len'],
        max_num_seqs=1,max_num_batched_tokens=ev['max_model_len'],gpu_memory_utilization=ev['gpu_memory_utilization'],enforce_eager=True,
        enable_prefix_caching=False,generation_config='vllm',logprobs_mode='raw_logprobs',seed=rc['seed'],
        enable_lora=True,max_lora_rank=8,max_loras=len(cfg['checkpoints']))
    rows=[]
    try:
        total=len(cfg['checkpoints'])*len(contexts)*(1+rc['samples_per_context'])
        with tqdm(total=total,desc='Recovery diagnostic',unit='action',file=sys.stdout,disable=False,dynamic_ncols=True,mininterval=.3) as progress:
            for bi,(branch,spec) in enumerate(cfg['checkpoints'].items(),1):
                request=LoRARequest(branch,bi,str(ROOT/'artifacts'/spec['run']/'final/adapter'))
                with (output/(branch+'-actions.jsonl')).open('x',encoding='utf8') as stream:
                    for c in contexts:
                        prompt=tok.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)
                        if len(prompt)+ev['max_new_tokens']>ev['max_model_len']:raise ValueError('diagnostic context budget exceeded')
                        for mode in ('greedy','sample'):
                            for ri in range(1 if mode=='greedy' else rc['samples_per_context']):
                                seed=sampling_seed(cfg,tasks,c,ri)
                                params=SamplingParams(n=1,temperature=0 if mode=='greedy' else rc['temperature'],top_p=1,top_k=-1,
                                    max_tokens=ev['max_new_tokens'],logprobs=1,stop_token_ids=[tok.eos_token_id],detokenize=False,seed=seed)
                                native=engine.generate([{'prompt_token_ids':prompt}],params,lora_request=request,use_tqdm=False)[0]
                                if list(native.prompt_token_ids)!=list(prompt):raise ValueError('native prompt differs')
                                completion=native.outputs[0];ids=list(completion.token_ids)
                                if not ids or completion.logprobs is None or len(ids)!=len(completion.logprobs):raise ValueError('native probability alignment required')
                                raw=[p[t].logprob for t,p in zip(ids,completion.logprobs)]
                                generation=asdict(Generation(tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False),
                                    'greedy' if mode=='greedy' else 'dpo_sample',list(prompt),ids,raw,None,completion.finish_reason))
                                validate_generation(tok,c['messages'],generation,generation['mode'])
                                actual,diagnostic=diagnose_action(tm[c['task_id']],c,generation['text'])
                                row={'model_branch':branch,'task_id':c['task_id'],'family_id':c['family_id'],'split':c['split'],'context_id':c['context_id'],
                                    'scenario':c['scenario'],'mode':mode,'sample_index':ri,'sample_seed':seed,
                                    'temperature':0 if mode=='greedy' else rc['temperature'],'generation':generation,'row':actual,'diagnostic':diagnostic}
                                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
                                progress.set_postfix(branch=branch,split=c['split'],scenario=c['scenario'],refresh=False);progress.update(1)
    finally:
        engine.llm_engine.engine_core.shutdown()
        import torch.distributed as dist
        if dist.is_initialized():dist.destroy_process_group()
    write(output/'diagnostic_summary.json',diagnostic_summary(rows,cfg['checkpoints']));verify_inputs(output)

def audit_diagnostics(output,cfg,tok):
    tasks=selected_tasks(cfg);tm={t.id:t for t in tasks};contexts=diagnostic_contexts(tasks)
    if contexts!=load(output/'contexts.jsonl'):raise ValueError('fixed diagnostic contexts differ')
    pairs=load(output/'pairs.jsonl');validate_training_pairs(pairs);seen={messages_id(p['messages']) for p in pairs}
    membership=[{'context_id':c['context_id'],'task_id':c['task_id'],'split':c['split'],'messages_sha256':messages_id(c['messages']),
        'seen_in_training_pairs':messages_id(c['messages']) in seen} for c in contexts]
    if membership!=load(output/'context_membership.jsonl') or any(m['seen_in_training_pairs']!=(m['split']=='train') for m in membership):
        raise ValueError('training context membership differs')
    cm={c['context_id']:c for c in contexts};rc=cfg['recovery_eval'];all_rows=[]
    expected={(c['context_id'],mode,i) for c in contexts for mode in ('greedy','sample') for i in range(1 if mode=='greedy' else rc['samples_per_context'])}
    for branch in cfg['checkpoints']:
        rows=load(output/(branch+'-actions.jsonl'))
        if len(rows)!=len(expected) or {(r['context_id'],r['mode'],r['sample_index']) for r in rows}!=expected:raise ValueError('complete distinct diagnostic actions required')
        for row in rows:
            c=cm[row['context_id']];mode=row['mode']
            if row['model_branch']!=branch or row['task_id']!=c['task_id'] or row['family_id']!=c['family_id'] or row['split']!=c['split'] or row['scenario']!=c['scenario'] or row['sample_seed']!=sampling_seed(cfg,tasks,c,row['sample_index']) or row['temperature']!=(0 if mode=='greedy' else rc['temperature']):
                raise ValueError('diagnostic action identity differs')
            g=row['generation'];validate_generation(tok,c['messages'],g,'greedy' if mode=='greedy' else 'dpo_sample')
            if len(g['response_token_ids'])>cfg['eval']['max_new_tokens']:raise ValueError('diagnostic token budget differs')
            if g['finish_reason']=='stop' and g['response_token_ids'][-1]!=tok.eos_token_id:raise ValueError('diagnostic EOS differs')
            actual,diagnostic=diagnose_action(tm[c['task_id']],c,g['text'])
            if actual!=row['row'] or diagnostic!=row['diagnostic']:raise ValueError('diagnostic replay differs')
        all_rows.extend(rows)
    summary=diagnostic_summary(all_rows,cfg['checkpoints'])
    if summary!=read(output/'diagnostic_summary.json'):raise ValueError('diagnostic summary differs')
    return {'passed':True,'actions_replayed':len(all_rows),'train_contexts':sum(c['split']=='train' for c in contexts),
        'development_contexts':sum(c['split']=='development' for c in contexts),'training_contexts_exactly_in_pairs':sum(m['seen_in_training_pairs'] for m in membership),
        'development_contexts_in_training_pairs':0,'training_updates':0,'test_split_used':False}

def export_csv(output):
    summary=read(output/'diagnostic_summary.json')
    columns=['branch','split','mode','scenario','actions','productive','tool_matches_expert','evidence_actions','correct_evidence_actions',
        'expected_tool_evidence_actions','expected_tool_correct_evidence_actions','evidence_slots','correct_evidence_slots','sql_attempts',
        'sql_any_metric_verified','query_required_actions','query_chosen_when_required','query_target_verified_when_required']
    with (output/'diagnostic_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for branch,splits in summary.items():
            for split,modes in splits.items():
                for mode,groups in modes.items():
                    for scenario,s in [('overall',groups['overall']),*groups['by_scenario'].items()]:
                        e=s['evidence'];q=s['sql']
                        writer.writerow(dict(zip(columns,[branch,split,mode,scenario,s['actions'],s['productive'],s['tool_matches_expert'],e['actions'],e['all_references_verified'],
                            e['expected_tool_actions'],e['expected_tool_all_references_verified'],e['slots'],e['slot_status_counts'].get('correct_verified_metric',0),
                            q['attempts'],q['any_metric_verified'],q['query_required_actions'],q['query_chosen_when_required'],q['target_verified_when_required']])))

def audit(output):
    cfg,identity=verify_inputs(output);binding=read(output/'input_binding.json')
    if file_hash(output/'contexts.jsonl')!=binding['contexts_sha256'] or file_hash(output/'context_membership.jsonl')!=binding['membership_sha256']:raise ValueError('context hash differs')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    evidence=audit_diagnostics(output,cfg,tok);export_csv(output);write(output/'train_development_comparison.json',comparison_rates(read(output/'diagnostic_summary.json')));verify_inputs(output);write(output/'audit.json',evidence)

if __name__=='__main__':
    import argparse
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['prepare','evaluate','audit'],required=True)
    args=parser.parse_args();globals()[args.mode](args.output)
