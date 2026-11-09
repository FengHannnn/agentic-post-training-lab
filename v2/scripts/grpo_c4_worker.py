"""C4 prepare, resumable single-adapter evaluation and audit worker."""
import argparse
import copy
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from grpo_c4_common import (ROOT,read,atomic_json,file_hash,verify_files,under,validate_config,
    runtime_identity,validate_runtime,driver_identity,verify_inputs,quality_specs,verify_session,
    verify_episode,verify_model,audit,LABELS,ADAPTERS,VERSIONS,confirmation_tasks,confirmation_plan,
    independence_report,summarize,expert_confirmation)
from apt_v2.grpo import run_continuation,digest
from apt_v2.dpo_recovery import SCENARIOS,build_context
from grpo_common import jsonl


def prepare(output):
    from tqdm.auto import tqdm
    cfg=validate_config(read(output/'config.json'));actual=runtime_identity();validate_runtime(cfg,actual)
    for name,value in [('execution_environment.json',actual),('driver_identity.json',driver_identity())]:
        if (output/name).exists() and read(output/name)!=value:raise ValueError('prepared C4 hardware identity differs')
        atomic_json(output/name,value)
    identity=read(under(ROOT,cfg['base_identity_file']))
    if identity['model_id']!=cfg['base_model'] or identity['resolved_revision']!=cfg['revision']:raise ValueError('fixed base revision differs')
    for name,sha in tqdm(identity['files_sha256'].items(),desc='C4 base identity',unit='file',file=sys.stdout,disable=False):
        if file_hash(under(ROOT,cfg['base_dir'])/name)!=sha:raise ValueError('C4 base model file differs: '+name)
    atomic_json(output/'model_identity.json',identity)
    for label in LABELS:
        source=under(ROOT,cfg['models'][label]['input_dir'])
        if file_hash(source/'adapter/adapter_model.safetensors')!=ADAPTERS[label]:raise ValueError('preselected candidate hash differs')
        shutil.copytree(source,output/'inputs'/label,dirs_exist_ok=True)
    tok=output/'tokenizer';tok.mkdir(exist_ok=True)
    for name in identity['files_sha256']:
        if not name.endswith('.safetensors'):shutil.copyfile(under(ROOT,cfg['base_dir'])/name,tok/name)
    atomic_json(output/'tasks.json',[asdict(t) for t in confirmation_tasks(cfg)]);atomic_json(output/'plan.json',confirmation_plan(cfg))
    atomic_json(output/'independence.json',independence_report(cfg))
    proof=[]
    for t in tqdm(confirmation_tasks(cfg),desc='C4 expert budget proof',unit='task',file=sys.stdout,disable=False):
        proof.extend(expert_confirmation(t,s) for s in (None,*SCENARIOS))
    atomic_json(output/'expert_budget_proof.json',proof)
    for src,dest in [(cfg['contract_report'],'contract.md'),(cfg['validation_report'],'local_validation.json')]:shutil.copyfile(under(ROOT,src),output/dest)
    names=['config.json','execution_environment.json','driver_identity.json','model_identity.json','tasks.json','plan.json','independence.json','expert_budget_proof.json','contract.md','local_validation.json']
    names += [p.relative_to(output).as_posix() for directory in (output/'inputs',tok) for p in sorted(directory.rglob('*')) if p.is_file()]
    atomic_json(output/'input_binding.json',{'files':{n:file_hash(output/n) for n in names},'config_sha256':file_hash(output/'config.json')})
    verify_inputs(output)


def evaluate(output,label):
    from tqdm.auto import tqdm
    from transformers import AutoTokenizer
    from grpo_vllm import engine_and_tokenizer,BehaviorPolicy,shutdown
    cfg=verify_inputs(output);target=output/'evaluations'/(label+'.json')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    if target.exists():verify_model(output,label,tok);return
    plan=confirmation_plan(cfg);tasks={t.id:t for t in confirmation_tasks(cfg)};session_dir=output/'sessions'/label;session_dir.mkdir(parents=True,exist_ok=True)
    episode_dir=output/'episodes'/label;episode_dir.mkdir(parents=True,exist_ok=True)
    sessions={p.stem:read(p) for p in sorted(session_dir.glob('*.json'))}
    for s in sessions.values():verify_session(output,label,s,tok,cfg)
    committed={}
    for p in sorted(episode_dir.glob('*.json')):
        if not p.stem.isdigit() or len(p.stem)!=4 or not 0<=int(p.stem)<len(plan):raise ValueError('unexpected committed C4 episode cursor')
        i=int(p.stem);e=read(p)
        if e['plan_index']!=i or e['session_id'] not in sessions:raise ValueError('committed C4 episode/session binding differs')
        verify_episode(tasks[plan[i]['task_id']],plan[i],e,label,tok,cfg,e['session_id'],digest(sessions[e['session_id']]['worker_proof']));committed[i]=e
    engine=None;started=time.monotonic()
    partial=output/'interruptions'/(label+'-'+str(time.time_ns())+'.jsonl');partial.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tqdm(total=len(plan)+14,desc='C4 Evaluate '+label,unit='episode/action',file=sys.stdout,disable=False,dynamic_ncols=True) as bar:
            bar.update(len(committed))
            if len(committed)<len(plan):
                engine_cfg=copy.deepcopy(cfg);engine_cfg['seed']=cfg['confirmation_seed']
                engine,tok,lora=engine_and_tokenizer(engine_cfg,output/'inputs'/label/'adapter');quality=[]
                for spec in quality_specs(cfg):
                    context=build_context(tasks[spec['task_id']],spec['scenario'],allowed_split='test');gens=[]
                    for _ in range(2):
                        policy=BehaviorPolicy(engine,tok,lora,{**cfg['sampling'],'temperature':0},spec['seed'])
                        gens.append(asdict(policy(context['messages'])));bar.set_postfix(kind='quality',mode='greedy',refresh=False);bar.update(1)
                    quality.append({**spec,'generations':gens})
                sid=str(time.time_ns());adapter=output/'inputs'/label/'adapter'
                session={'session_id':sid,'label':label,'config_sha256':file_hash(output/'config.json'),'adapter_sha256':ADAPTERS[label],
                    'hardware':runtime_identity(),'driver':driver_identity(),'quality':quality,
                    'worker_proof':engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),ADAPTERS[label]),timeout=60),
                    'one_adapter_one_process':True,'max_loras':1,'lora_id':1}
                verify_session(output,label,session,tok,cfg);atomic_json(session_dir/(sid+'.json'),session);sessions[sid]=session
                for i,item in enumerate(plan):
                    if i in committed:continue
                    sampling={**cfg['sampling'],'temperature':item['temperature']}
                    policy=BehaviorPolicy(engine,tok,lora,sampling,item['seed'],lambda n:bar.set_postfix(kind=item['kind'],mode=item['mode'],turn=policy.calls,response_tokens=n,refresh=True))
                    try:e=run_continuation(tasks[item['task_id']],item['scenario'],policy,sampling,VERSIONS[label],item['seed'],item['rollout_index'],cfg['tool_cost'])
                    except Exception as exc:
                        jsonl(partial,{**getattr(exc,'episode_evidence',{'error':str(exc)}),'label':label,'plan_index':i,'session_id':sid});raise
                    episode_proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),ADAPTERS[label]),timeout=60)
                    if episode_proof!=session['worker_proof']:raise ValueError('episode loaded tensor proof differs')
                    e.update(model_branch=label,evaluation_mode=item['mode'],plan_index=i,session_id=sid,worker_proof_sha256=digest(episode_proof))
                    verify_episode(tasks[item['task_id']],item,e,label,tok,cfg,sid,digest(session['worker_proof']));atomic_json(episode_dir/f'{i:04}.json',e);committed[i]=e
                    bar.set_postfix(kind=item['kind'],mode=item['mode'],success=e['success'],refresh=False);bar.update(1)
                proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),ADAPTERS[label]),timeout=60)
                if proof!=session['worker_proof']:raise ValueError('end-of-session loaded tensor proof differs')
            else:bar.update(14)
        episodes=[committed[i] for i in range(len(plan))]
        data={'label':label,'config_sha256':file_hash(output/'config.json'),'episodes':episodes,
              'sessions':list(sessions.values()),'summary':summarize(episodes,cfg),'elapsed_seconds_this_stage':time.monotonic()-started,
              'resume_unit':'verified complete episode','new_training_updates':0}
        check=verify_model(output,label,tok,data=data);atomic_json(output/'evaluations'/(label+'-audit.json'),check);atomic_json(target,data)
    finally:
        if engine is not None:shutdown(engine)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate','audit']);p.add_argument('--output',type=Path,required=True);p.add_argument('--label',choices=LABELS);p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.output)
    elif a.command=='evaluate':
        if not a.label:p.error('--label required for evaluate')
        evaluate(a.output,a.label)
    else:audit(a.output,a.local)
