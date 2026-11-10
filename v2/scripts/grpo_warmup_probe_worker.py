"""Immutable single-action train probes and native teacher-forced alias diagnostics."""
import argparse,copy,json,math,shutil,sys,time
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,runtime_identity,validate_runtime,jsonl
from apt_v2.grpo import digest
from apt_v2.grpo_warmup_probe import (validate_config,validate_prior,make_plan,execute,variants,validate_row,
                                      score_summary,summarize,MODEL_IDENTITIES)


def model_identity(cfg,folder,name):
    meta=read(folder/'complete.json');sha,count=MODEL_IDENTITIES[name]
    if (not {'adapter_config.json','adapter_model.safetensors'}<=set(meta['adapter_sha256'])
            or set(meta['adapter_sha256'])-{'adapter_config.json','adapter_model.safetensors','README.md'}
            or meta['total_sft_updates']!=count or meta['total_dpo_updates']!=11 or meta.get('total_rl_updates',0)!=0
            or file_hash(folder/'adapter/adapter_model.safetensors')!=sha or not read(folder/'source_audit.json')['passed']):
        raise ValueError('probe model identity differs: '+name)
    if name!='w' and (meta.get('parent_W_sha256')!=MODEL_IDENTITIES['w'][0]
                      or not read(folder/'source_audit.json').get('train_only')):
        raise ValueError('probe B ancestry differs')
    verify_files(folder/'adapter',meta['adapter_sha256'])
    return read(folder/'model_identity.json')


def worker_proof(proof,folder,sha):
    from grpo_vllm import projected_checkpoint_tensors
    projected=projected_checkpoint_tensors(folder/'adapter')
    if not proof or any(not p['passed'] or p['adapter_sha256']!=sha or p['tensor_count']!=112
                        or p['registered_ids']!=[1] or 1 not in p['active_slots'] or p['tensors']!=projected
                        or p['loaded_tensor_sha256']!=digest(projected) for p in proof):
        raise ValueError('probe worker loaded tensors differ')
    return True


def verify_inputs(output,local=False):
    cfg=read(output/'config.json');validate_config(cfg);manifest=read(output/'run_manifest.json')
    verify_files(ROOT,manifest['source_sha256']);verify_files(output/'source',manifest['source_sha256'])
    if file_hash(output/'release_manifest.json')!=manifest['release_sha256']:raise ValueError('publication snapshot differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files'])
    prepared=output/'prepared';binding=read(prepared/'manifest.json');verify_files(prepared,binding['files'])
    if binding['config_sha256']!=file_hash(output/'config.json'):raise ValueError('prepared config differs')
    prior=read(prepared/'inputs/prior-evidence.json');validate_prior(prior,cfg)
    if prior!=read(ROOT/cfg['input_dir']/'prior-evidence.json'):raise ValueError('prior snapshot differs')
    data=read(prepared/'inputs/sft-examples.json');plan=make_plan(data)
    if read(prepared/'inputs/probe-plan.json')!=plan or read(ROOT/cfg['input_dir']/'probe-plan.json')!=plan:
        raise ValueError('frozen train probe plan differs')
    branch=read(ROOT/cfg['branch_template'])
    if read(prepared/'branch_config.json')!=branch:raise ValueError('base inference config differs')
    for name in cfg['models']:
        identity=model_identity(cfg,prepared/'inputs/models'/name,name)
        if identity['model_id']!=branch['base_model'] or identity['resolved_revision']!=branch['revision']:
            raise ValueError('model base revision differs')
        for n,sha in identity['files_sha256'].items():
            if not n.endswith('.safetensors') and file_hash(prepared/'tokenizer'/n)!=sha:
                raise ValueError('probe tokenizer identity differs')
    if not local:
        actual=runtime_identity();validate_runtime(branch,actual)
        if actual!=read(output/'execution_environment.json'):raise ValueError('probe runtime differs')
    return cfg,plan


def prepare(output):
    cfg=read(output/'config.json');validate_config(cfg)
    if (output/'prepared').exists():return verify_inputs(output)
    branch=read(ROOT/cfg['branch_template']);validate_runtime(branch,runtime_identity())
    source=ROOT/cfg['input_dir'];validate_prior(read(source/'prior-evidence.json'),cfg)
    identities=[model_identity(cfg,source/'models'/name,name) for name in cfg['models']]
    if any(i!=identities[0] for i in identities):raise ValueError('all three models require the same base/tokenizer')
    staging=output/('.prepare-'+str(time.time_ns()));shutil.copytree(source,staging/'inputs')
    identity=identities[0];(staging/'tokenizer').mkdir()
    for n,sha in tqdm(identity['files_sha256'].items(),desc='Prepare probe base',unit='file',file=sys.stdout,disable=False):
        if file_hash(ROOT/branch['base_dir']/n)!=sha:raise ValueError('cached base identity differs: '+n)
        if not n.endswith('.safetensors'):shutil.copyfile(ROOT/branch['base_dir']/n,staging/'tokenizer'/n)
    atomic_json(staging/'branch_config.json',branch)
    atomic_json(staging/'manifest.json',{'config_sha256':file_hash(output/'config.json'),
        'files':{p.relative_to(staging).as_posix():file_hash(p) for p in sorted(staging.rglob('*')) if p.is_file()}})
    staging.rename(output/'prepared');return verify_inputs(output)


def probe_inputs(output,local,protocol):
    return verify_inputs(output,local=local) if protocol is None else protocol.verify_probe_inputs(output,local=local)


def probe_folder(output,name,protocol):
    return output/'prepared/inputs/models'/name if protocol is None else protocol.probe_model_folder(output,name)


def verify_model(output,name,tok,require_complete=True,protocol=None):
    cfg,plan=probe_inputs(output,True,protocol);root=output/'probes'/name;folder=probe_folder(output,name,protocol)
    sessions={p.stem:read(p) for p in (root/'sessions').glob('*.json')};rows=[]
    for sid,s in sessions.items():
        if (s['session_id']!=sid or s['model']!=name or s['config_sha256']!=file_hash(output/'config.json')
                or s['prepared_sha256']!=file_hash(output/'prepared/manifest.json')
                or s['hardware']!=read(output/'execution_environment.json') or s['adapter_sha256']!=cfg['models'][name]['adapter_sha256']):
            raise ValueError('probe session identity differs')
        worker_proof(s['worker_proof'],folder,s['adapter_sha256'])
    for p in sorted((root/'rows').glob('*.json')):
        r=read(p);index=r['index']
        if type(index) is not int or not 0<=index<24 or p.name!=f'{index:04}.json' or r['model']!=name or r['session_id'] not in sessions:
            raise ValueError('probe row identity differs')
        validate_row(cfg,plan[index],r,tok);rows.append(r)
    if require_complete:
        result=summarize(plan,rows);complete=read(root/'complete.json');verify_files(root,complete['files'])
        if complete['config_sha256']!=file_hash(output/'config.json') or complete['summary']!=result:
            raise ValueError('probe committed summary differs')
        return result
    return rows


def forced_score(engine,lora,prompt,response,seed):
    from vllm import SamplingParams
    params=SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1,prompt_logprobs=1,detokenize=False,seed=seed)
    result=engine.generate([{'prompt_token_ids':prompt+response}],params,lora_request=lora,use_tqdm=False)[0]
    if list(result.prompt_token_ids)!=prompt+response or len(result.outputs)!=1:raise ValueError('forced prompt differs')
    logps=[float(result.prompt_logprobs[len(prompt)+j][token].logprob) for j,token in enumerate(response)]
    return {'source':'vllm_prompt_logprobs_T1','not_used_for_policy_updates':True,'prompt_sha256':digest(prompt+response),
            'response_token_ids':response,'token_log_probs':logps,'auxiliary_token_ids':list(result.outputs[0].token_ids),'seed':seed}


def probe_model(output,name,protocol=None):
    from transformers import AutoTokenizer
    from grpo_vllm import engine_and_tokenizer,BehaviorPolicy,shutdown
    cfg,plan=probe_inputs(output,False,protocol);prepared=output/'prepared';root=output/'probes'/name
    tok=AutoTokenizer.from_pretrained(prepared/'tokenizer',local_files_only=True,trust_remote_code=False)
    if (root/'complete.json').exists():return verify_model(output,name,tok,protocol=protocol)
    committed=verify_model(output,name,tok,require_complete=False,protocol=protocol);done={r['index'] for r in committed}
    branch=read(prepared/'branch_config.json');branch['seed']=cfg['seed'];sampling=dict(cfg['sampling'],temperature=0)
    folder=probe_folder(output,name,protocol);adapter=folder/'adapter';sha=cfg['models'][name]['adapter_sha256']
    engine=None;sid=str(time.time_ns());session=None;current=None
    (root/'attempts').mkdir(parents=True,exist_ok=True)
    try:
        with tqdm(total=24,initial=len(done),desc='Probe '+name,unit='context',file=sys.stdout,disable=False) as bar:
            if len(done)<24:engine,tok,lora=engine_and_tokenizer(branch,adapter)
            for item in plan:
                if item['index'] in done:continue
                seed=cfg['seed']+item['index']*100
                current={'index':item['index'],'task_id':item['task_id'],'state':item['state'],'model':name,'seed':seed,
                         'plan_sha256':digest(item),'session_id':sid,'generations':[],'executions':[],
                         'scores':None,'score_summary':None}
                for ri in range(2 if item['kind']=='target' else 1):
                    policy=BehaviorPolicy(engine,tok,lora,sampling,seed,
                        lambda n:bar.set_postfix(kind=item['kind'],phase='greedy',tokens=n,refresh=True))
                    generation=asdict(policy(item['messages']));current['generations'].append(generation)
                    current['executions'].append(execute(item,generation['text']))
                    jsonl(root/'attempts'/(sid+'-requests.jsonl'),{'index':item['index'],'phase':'greedy','repeat':ri,'generation':generation})
                proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),sha),timeout=60)
                worker_proof(proof,folder,sha)
                if session is None:
                    session={'session_id':sid,'model':name,'config_sha256':file_hash(output/'config.json'),
                             'prepared_sha256':file_hash(prepared/'manifest.json'),'adapter_sha256':sha,
                             'hardware':read(output/'execution_environment.json'),'worker_proof':proof}
                    atomic_json(root/'sessions'/(sid+'.json'),session)
                elif proof!=session['worker_proof']:raise ValueError('probe model changed during session')
                if item['kind']=='target':
                    v=variants(tok,item);scores={}
                    for offset,(key,ids) in enumerate((('correct',v['correct_ids']),('omitted',v['omitted_ids']))):
                        bar.set_postfix(kind='target',phase=key+'-prefill',refresh=True)
                        scores[key]=forced_score(engine,lora,v['prompt_ids'],ids,seed+17+offset)
                        jsonl(root/'attempts'/(sid+'-requests.jsonl'),{'index':item['index'],'phase':key,'score':scores[key]})
                    current['scores']=scores;current['score_summary']=score_summary(v,scores)
                validate_row(cfg,item,current,tok);atomic_json(root/'rows'/f"{item['index']:04}.json",current)
                committed.append(current);current=None;bar.update(1)
        if engine is not None:
            final_proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),sha),timeout=60)
            if final_proof!=session['worker_proof']:raise ValueError('final probe worker differs')
        summary=summarize(plan,committed)
        files={p.relative_to(root).as_posix():file_hash(p) for directory in ('rows','sessions','attempts','interruptions') for p in sorted((root/directory).glob('*')) if p.is_file()}
        atomic_json(root/'complete.json',{'config_sha256':file_hash(output/'config.json'),'files':files,'summary':summary})
        return verify_model(output,name,tok,protocol=protocol)
    except BaseException as exc:
        atomic_json(root/'interruptions'/(sid+'.json'),{'error':str(exc),'partial':current});raise
    finally:
        if engine is not None:shutdown(engine)


def request_accounting(output):
    counts={'logged_greedy_requests':0,'logged_forced_requests':0,'logged_auxiliary_tokens':0,'truncated_log_lines':[]}
    for path in sorted((output/'probes').glob('*/attempts/*.jsonl')):
        for i,line in enumerate(path.read_text(encoding='utf8').splitlines()):
            try:row=json.loads(line)
            except (ValueError,TypeError):
                counts['truncated_log_lines'].append({'file':path.relative_to(output).as_posix(),'line':i+1});continue
            if row['phase']=='greedy':counts['logged_greedy_requests']+=1
            else:
                counts['logged_forced_requests']+=1;counts['logged_auxiliary_tokens']+=len(row['score']['auxiliary_token_ids'])
    return counts


def audit(output,local=False):
    from transformers import AutoTokenizer
    cfg,plan=verify_inputs(output,local);tok=AutoTokenizer.from_pretrained(output/'prepared/tokenizer',local_files_only=True,trust_remote_code=False)
    models={}
    for name in tqdm(cfg['models'],desc='Replay warm-up probes',unit='model',file=sys.stdout,disable=False):models[name]=verify_model(output,name,tok)
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_native_diagnostic','models':models,
            'committed_generated_actions':108,'committed_forced_score_requests':72,'committed_auxiliary_generated_tokens':72,
            'request_accounting':request_accounting(output),'actual_sft_updates':0,'actual_rl_updates':0}
    if local:
        cloud=read(output/'audit.json')
        if {k:v for k,v in cloud.items() if k!='scope'}!={k:v for k,v in result.items() if k!='scope'}:
            raise ValueError('cloud/local probe summary differs')
    atomic_json(output/('local_audit.json' if local else 'audit.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','model','audit']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=tuple(MODEL_IDENTITIES));p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage=='model':
        if a.model is None:p.error('--model required')
        probe_model(a.output,a.model)
    else:audit(a.output,a.local)
