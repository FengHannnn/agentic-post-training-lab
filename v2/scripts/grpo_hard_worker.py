"""Frozen W collection with per-episode transactions, progress and paired signal audit."""
import argparse,shutil,sys,time
from pathlib import Path
from dataclasses import asdict
from grpo_hard_common import (ROOT,read,atomic_json,file_hash,verify_files,under,validate_config,runtime_identity,
    validate_runtime,driver_identity,verify_inputs,verify_session,verify_episode,verify_result,audit,W_SHA)
from apt_v2.grpo_hard_preflight import tasks,plan,context,frozen_inputs,summarize
from apt_v2.grpo import run_continuation,digest
from grpo_common import jsonl


def prepare(output):
    from tqdm.auto import tqdm
    cfg=validate_config(read(output/'config.json'));actual=runtime_identity();validate_runtime(cfg,actual)
    for name,value in [('execution_environment.json',actual),('driver_identity.json',driver_identity())]:
        if (output/name).exists() and read(output/name)!=value:raise ValueError('prepared preflight hardware identity differs')
        atomic_json(output/name,value)
    initial=under(ROOT,cfg['initial_dir']);identity=read(initial/'model_identity.json');complete=read(initial/'complete.json')
    if (identity['model_id']!=cfg['base_model'] or identity['resolved_revision']!=cfg['revision']
            or complete['total_sft_updates']!=513 or complete['total_dpo_updates']!=11
            or file_hash(initial/'adapter/adapter_model.safetensors')!=W_SHA):raise ValueError('frozen W ancestry differs')
    verify_files(initial/'adapter',complete['adapter_sha256'])
    for name,sha in tqdm(identity['files_sha256'].items(),desc='Prepare frozen base',unit='file',file=sys.stdout,disable=False):
        if file_hash(under(ROOT,cfg['base_dir'])/name)!=sha:raise ValueError('frozen base file differs: '+name)
    shutil.copytree(initial,output/'initial',dirs_exist_ok=True);atomic_json(output/'model_identity.json',identity)
    tok=output/'tokenizer';tok.mkdir(exist_ok=True)
    for name in identity['files_sha256']:
        if not name.endswith('.safetensors'):shutil.copyfile(under(ROOT,cfg['base_dir'])/name,tok/name)
    for name,value in frozen_inputs(cfg).items():atomic_json(output/name,value)
    for src,dest in [(cfg['contract_report'],'contract.md'),(cfg['validation_report'],'local_validation.json')]:shutil.copyfile(under(ROOT,src),output/dest)
    names=['config.json','execution_environment.json','driver_identity.json','model_identity.json','contract.md','local_validation.json',*frozen_inputs(cfg)]
    names += [p.relative_to(output).as_posix() for folder in (output/'initial',tok) for p in folder.rglob('*') if p.is_file()]
    atomic_json(output/'input_binding.json',{'config_sha256':file_hash(output/'config.json'),'files':{n:file_hash(output/n) for n in names}})
    verify_inputs(output)


def collect(output):
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    from grpo_vllm import engine_and_tokenizer,BehaviorPolicy,shutdown
    cfg=verify_inputs(output);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    if (output/'results.json').exists():return verify_result(output,tok)
    expected=plan(cfg);task_map={t.id:t for t in tasks()};committed={};sessions={}
    for p in sorted((output/'sessions').glob('*.json')):
        s=read(p)
        if p.stem!=s['session_id']:raise ValueError('session file identity differs')
        verify_session(output,s,tok,cfg);sessions[s['session_id']]=s
    for p in sorted((output/'episodes').glob('*.json')):
        if not p.stem.isdigit() or len(p.stem)!=4 or not 0<=int(p.stem)<96:raise ValueError('valid committed episode cursor required')
        i=int(p.stem);e=read(p)
        if e['session_id'] not in sessions:raise ValueError('committed episode has no native session proof')
        verify_episode(output,expected[i],e,tok,cfg,sessions[e['session_id']]);committed[i]=e
    engine=None;start=time.monotonic();sid=str(time.time_ns());adapter=output/'initial/adapter'
    partial=output/'interruptions'/(sid+'.jsonl');partial.parent.mkdir(exist_ok=True)
    try:
        with tqdm(total=96,desc='Paired hard recovery',unit='episode',file=sys.stdout,disable=False,dynamic_ncols=True) as bar:
            bar.update(len(committed))
            if len(committed)<96:
                first=next(i for i in range(96) if i not in committed)
                setup={'seed':expected[first]['seed'],'policy_sha256':W_SHA,'token_limit':16,'sampling':cfg['sampling']}
                engine,tok,lora=engine_and_tokenizer(cfg,adapter,setup);session=None
                for item in expected:
                    i=item['index']
                    if i in committed:continue
                    def show(n):bar.set_postfix(scenario=item['base_scenario'],level=item['level'],turn=policy.calls,tokens=n,refresh=True)
                    policy=BehaviorPolicy(engine,tok,lora,cfg['sampling'],item['seed'],show)
                    try:e=run_continuation(task_map[item['task_id']],item['scenario'],policy,cfg['sampling'],0,item['seed'],item['rollout_index'],cfg['tool_cost'],context)
                    except Exception as exc:
                        jsonl(partial,{**getattr(exc,'episode_evidence',{'error':str(exc)}),'plan_index':i,'session_id':sid});raise
                    proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),W_SHA),timeout=60)
                    if session is None:
                        session={'session_id':sid,'first_plan_index':i,'first_generation':asdict(policy.first),
                            'config_sha256':file_hash(output/'config.json'),'adapter_sha256':W_SHA,'hardware':runtime_identity(),
                            'driver':driver_identity(),'worker_proof':proof,'new_training_updates':0,'max_loras':1,'lora_id':1}
                        verify_session(output,session,tok,cfg);atomic_json(output/'sessions'/(sid+'.json'),session);sessions[sid]=session
                    elif proof!=session['worker_proof']:raise ValueError('W tensors or native probability capture changed')
                    e.update(model_branch='weighted_sft',base_scenario=item['base_scenario'],level=item['level'],plan_index=i,
                             session_id=sid,worker_proof_sha256=digest(proof))
                    verify_episode(output,item,e,tok,cfg,session);atomic_json(output/'episodes'/f'{i:04}.json',e);committed[i]=e
                    bar.set_postfix(scenario=item['base_scenario'],level=item['level'],success=e['success'],progress=sum(len(r['new_verified_metrics']) for r in e['turns']),refresh=False);bar.update(1)
                end=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),W_SHA),timeout=60)
                if end!=session['worker_proof']:raise ValueError('end-of-session W tensor proof differs')
            rows=[committed[i] for i in range(96)];data={'config_sha256':file_hash(output/'config.json'),'episodes':rows,
                'sessions':list(sessions.values()),'summary':summarize(rows),'elapsed_seconds_this_attempt':time.monotonic()-start,
                'new_training_updates':0,'resume_unit':'verified complete episode'}
            check=verify_result(output,tok,data=data);atomic_json(output/'collection_audit.json',check);atomic_json(output/'results.json',data)
    finally:
        if engine is not None:shutdown(engine)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','audit']);p.add_argument('--output',type=Path,required=True);p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare(a.output)
    elif a.stage=='collect':collect(a.output)
    else:audit(a.output,a.local)
