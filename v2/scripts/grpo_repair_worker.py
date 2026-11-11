"""Repair study preparation, immutable W diagnosis and resumable supervised warm start."""
import argparse, copy, json, math, random, shutil, sys, time
from dataclasses import asdict
from pathlib import Path
from grpo_common import ROOT, read, atomic_json, file_hash, verify_files, verify_inputs, prepare, under, initial_policy
from apt_v2.grpo import initial_state, run_continuation, replay, digest, validate_native
from apt_v2.grpo_credit import LOCAL, compute_credit
from apt_v2.grpo_repair_study import (context, training_tasks, diagnostic_tasks, diagnostic_plan,
                                      route_diagnostic, COMPOUND)


def prepare_child(output, name):
    config=read(output/'branch-configs'/(name+'.json'));child=output/'branches'/name
    if child.exists():
        if read(child/'config.json')!=config:raise ValueError('child config differs')
        verify_inputs(child);return
    staging=output/'branches'/('.'+name+'-preparing-'+str(time.time_ns()))
    atomic_json(staging/'config.json',config)
    shutil.copyfile(output/'run_manifest.json',staging/'run_manifest.json')
    shutil.copytree(output/'source',staging/'source')
    prepare(staging)
    staging.rename(child)


def verify_diagnostic(output,tok):
    from grpo_vllm import verify_probability_probe, projected_checkpoint_tensors
    cfg=read(output/'config.json');child=output/'branches/w';branch=verify_inputs(child,runtime=False,local=True)
    plan=diagnostic_plan(cfg);tasks={t.id:t for t in diagnostic_tasks()};groups=[]
    sessions={p.stem:read(p) for p in (output/'diagnostic/sessions').glob('*.json')}
    projected=projected_checkpoint_tensors(child/'initial/adapter')
    for sid,session in sessions.items():
        if (session['session_id']!=sid or session['config_sha256']!=file_hash(output/'config.json')
                or session['adapter_sha256']!=branch['adapter_sha256']
                or session['hardware']!=read(child/'execution_environment.json')):
            raise ValueError('diagnostic session binding differs')
        proof=session['worker_proof']
        if not proof or any(not p['passed'] or p['adapter_sha256']!=branch['adapter_sha256'] or p['registered_ids']!=[1] or 1 not in p['active_slots']
                or p['tensor_count']!=112 or p['tensors']!=projected
                or p['loaded_tensor_sha256']!=digest(projected) for p in proof):
            raise ValueError('diagnostic worker tensors differ')
        verify_probability_probe(session['probability_probe'],session['first_generation'],branch,session['seed'],proof)
    from tqdm import tqdm
    for spec in tqdm(plan,desc='Replay diagnostic groups',unit='group',file=sys.stdout,disable=False):
        episodes=[]
        for ri in range(4):
            i=spec['index']*4+ri;e=read(output/'diagnostic/episodes'/f'{i:04}.json')
            if (e['task_id']!=spec['task_id'] or e['scenario']!=spec['scenario'] or e['seed']!=spec['seed']+ri*1000
                    or e['policy_version']!=0 or e['rollout_index']!=ri or e['sampling']!=branch['sampling']
                    or e.get('session_id') not in sessions):raise ValueError('diagnostic episode plan differs')
            replay(tasks[e['task_id']],e,tok,context);episodes.append(e)
        g={'spec':spec,'episodes':episodes,'credit':compute_credit(episodes,4,1e-8,LOCAL)}
        groups.append(g)
    result=route_diagnostic(groups)
    if (output/'diagnostic/route.json').exists() and result!=read(output/'diagnostic/route.json'):
        raise ValueError('recomputed diagnostic route differs')
    return groups,result


def diagnose(output):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from grpo_vllm import engine_and_tokenizer,BehaviorPolicy,probability_probe,shutdown
    cfg=read(output/'config.json');child=output/'branches/w';branch=verify_inputs(child)
    tok=AutoTokenizer.from_pretrained(child/'tokenizer',local_files_only=True,trust_remote_code=False)
    if (output/'diagnostic/route.json').exists():return verify_diagnostic(output,tok)[1]
    plan=diagnostic_plan(cfg);tasks={t.id:t for t in diagnostic_tasks()};adapter=child/'initial/adapter'
    items=[(spec,ri,spec['index']*4+ri) for spec in plan for ri in range(4)]
    missing=[x for x in items if not (output/'diagnostic/episodes'/f'{x[2]:04}.json').exists()]
    engine=None;sid=str(time.time_ns());session=None
    try:
        with tqdm(total=96,initial=96-len(missing),desc='Frozen W diagnostic',unit='episode',file=sys.stdout,disable=False) as bar:
            if missing:
                spec,ri,_=missing[0];seed=spec['seed']+ri*1000
                setup={'seed':seed,'policy_sha256':branch['adapter_sha256'],'token_limit':16,'sampling':branch['sampling']}
                engine,tok,lora=engine_and_tokenizer(branch,adapter,setup)
                for spec,ri,i in missing:
                    seed=spec['seed']+ri*1000
                    policy=BehaviorPolicy(engine,tok,lora,branch['sampling'],seed,
                        lambda n:bar.set_postfix(condition=spec['scenario'],turn=policy.calls,tokens=n,refresh=True))
                    try:e=run_continuation(tasks[spec['task_id']],spec['scenario'],policy,branch['sampling'],0,seed,ri,.01,context)
                    except Exception as exc:
                        atomic_json(output/'diagnostic/interruptions'/(str(time.time_ns())+'.json'),
                                    getattr(exc,'episode_evidence',{'error':str(exc)}));raise
                    replay(tasks[spec['task_id']],e,tok,context)
                    proof=engine.collective_rpc('grpo_verify_lora_tensors',args=(str(adapter.resolve()),branch['adapter_sha256']),timeout=60)
                    if session is None:
                        probe=probability_probe(engine,tok,lora,policy.first,branch,seed,worker_proof=proof)
                        if not probe['passed']:raise ValueError('diagnostic same-forward q failed')
                        session={'session_id':sid,'seed':seed,'worker_proof':proof,'probability_probe':probe,
                                 'first_generation':asdict(policy.first),'hardware':read(child/'execution_environment.json'),
                                 'adapter_sha256':branch['adapter_sha256'],'config_sha256':file_hash(output/'config.json')}
                        atomic_json(output/'diagnostic/sessions'/(sid+'.json'),session)
                    elif proof!=session['worker_proof']:raise ValueError('W session identity changed')
                    e['session_id']=sid;atomic_json(output/'diagnostic/episodes'/f'{i:04}.json',e);bar.update(1)
        groups,result=verify_diagnostic(output,tok)
        for g in groups:atomic_json(output/'diagnostic/groups'/f"{g['spec']['index']:04}.json",g)
        atomic_json(output/'diagnostic/route.json',result)
        return result
    finally:
        if engine is not None:shutdown(engine)


def supervision_dataset():
    from apt_v2.dpo import corrective_action
    examples=[];maintenance={k:[] for k in ('query_sql','calculate_change','check_result','submit')}
    def sample(task,messages,action,row,role):
        return {'task_id':task.id,'split':'train','role':role,'messages':copy.deepcopy(messages),
                'response':json.dumps(action),'observation':row['observation']}
    for task in training_tasks():
        env,messages,_,_=initial_state(task,COMPOUND,context_builder=context)
        try:
            action=corrective_action(env);row=env.step(action)
            if not row['new_verified_metrics'] or row['tool_error']:raise ValueError('productive first repair required')
            examples.append(sample(task,messages,action,row,'target'))
        finally:env.close()
        env,messages,_,_=initial_state(task,None,context_builder=context)
        try:
            while not env.done:
                action=corrective_action(env);row=env.step(action);tool=action['tool']
                if tool in maintenance and len(maintenance[tool])<3 and not row['tool_error'] and row['observation'].get('passed',True):
                    maintenance[tool].append(sample(task,messages,action,row,'maintenance'))
                messages.extend([{'role':'assistant','content':json.dumps(action)},
                                 {'role':'user','content':json.dumps({'observation':row['observation']})}])
        finally:env.close()
    examples += [e for rows in maintenance.values() for e in rows]
    if len(examples)!=24 or any(len(rows)!=3 for rows in maintenance.values()):
        raise ValueError('12 target and 12 stratified maintenance actions required')
    verify_supervision(examples)
    return examples


def verify_supervision(examples,expected_count=24):
    from apt_v2.environment import Environment
    from apt_v2.runtime import SYSTEM,parse_action
    tasks={t.id:t for t in training_tasks()}
    if len(examples)!=expected_count or sum(e['role']=='target' for e in examples)!=12:
        raise ValueError('frozen supervision composition differs')
    for e in examples:
        if e['split']!='train' or e['task_id'] not in tasks:raise ValueError('train-only supervision required')
        task=tasks[e['task_id']];messages=e['messages'];env=Environment(task)
        try:
            if messages[:2]!=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}]:
                raise ValueError('public prompt provenance differs')
            for i in range(2,len(messages),2):
                if messages[i]['role']!='assistant' or messages[i+1]['role']!='user':raise ValueError('tool prompt roles differ')
                row=env.step(parse_action(messages[i]['content']))
                if json.loads(messages[i+1]['content'])!={'observation':row['observation']}:raise ValueError('supervision prefix replay differs')
            row=env.step(parse_action(e['response']))
            if row['tool_error'] or row['observation']!=e['observation'] or not row['observation'].get('passed',True):
                raise ValueError('supervised response execution failed')
        finally:env.close()
    return True


def supervision_for(cfg):
    if cfg.get('schema') in ('repair_retention_v1','repair_dose_warm_v1','repair_lr_warm_v1'):
        from apt_v2.grpo_retention_study import supervision_dataset as retention_dataset
        return retention_dataset()
    return supervision_dataset()


def verify_for(cfg,data):
    if cfg.get('schema')=='repair_lr_warm_v1':
        from apt_v2.grpo_lr_study import validate_warm
        from apt_v2.grpo_retention_study import verify_dataset
        validate_warm(cfg)
        verify_dataset(data)
        return verify_supervision(data, expected_count=60)
    if cfg.get('schema')=='repair_dose_warm_v1':
        from apt_v2.grpo_dose_study import validate_warm
        from apt_v2.grpo_retention_study import verify_dataset
        validate_warm(cfg);verify_dataset(data)
        return verify_supervision(data,expected_count=60)
    if cfg.get('schema')=='repair_retention_v1':
        from apt_v2.grpo_retention_study import validate_master,verify_dataset
        validate_master(cfg);verify_dataset(data)
        return verify_supervision(data,expected_count=60)
    return verify_supervision(data)


def sft_order(cfg):
    if cfg.get('schema')=='repair_lr_warm_v1':
        from apt_v2.grpo_lr_study import order
        return order(cfg, read(ROOT/cfg['sft_examples']), read(ROOT/cfg['input_dir']/'order-r16.json'))
    if cfg.get('schema')=='repair_dose_warm_v1':
        from apt_v2.grpo_dose_study import order
        return order(cfg,read(ROOT/cfg['sft_examples']))
    if cfg.get('schema')=='repair_retention_v1':
        from apt_v2.grpo_retention_study import order
        return order(cfg)
    rng=random.Random(cfg['sft']['seed']);order=[]
    for _ in range(4):
        epoch=list(range(24));rng.shuffle(epoch);order.extend(epoch)
    return [order[i:i+4] for i in range(0,96,4)]


def sft_batch(model,optimizer,samples):
    import torch
    from apt_v2.sft import action_loss
    from grpo_worker import tensor_digest
    from peft import get_peft_model_state_dict
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    before=tensor_digest(get_peft_model_state_dict(model,adapter_name='policy'));losses=[]
    for sample in samples:
        loss=action_loss(model,sample);value=float(loss.detach())
        if not math.isfinite(value):raise ValueError('nonfinite SFT loss')
        (loss/len(samples)).backward();losses.append(value)
    norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
    if not math.isfinite(norm):raise ValueError('nonfinite SFT gradient')
    optimizer.step()
    after=tensor_digest(get_peft_model_state_dict(model,adapter_name='policy'))
    if before==after:raise ValueError('SFT weights unchanged')
    return {'losses':losses,'gradient_norm':norm,'before_tensor_sha256':before,'after_tensor_sha256':after,
            'optimizer_steps':sorted({int(v['step']) for v in optimizer.state.values()}),
            'frozen_parameter_gradients':sum(p.grad is not None for p in model.parameters() if not p.requires_grad)}


def warm_start(output):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel,get_peft_model_state_dict
    from tqdm import tqdm
    from grpo_worker import save_policy_adapter,rng_state,restore_rng,configure_training_memory,tensor_digest
    from grpo_common import checkpoint_binding
    from apt_v2.sft import encode_example,action_loss
    cfg=read(output/'config.json');child=output/'branches/w';branch=verify_inputs(child)
    data=supervision_for(cfg);steps=cfg['sft']['steps'];exposures=steps*cfg['sft']['batch_size']
    if read(ROOT/cfg['sft_examples'])!=data:raise ValueError('published SFT examples differ')
    data_path=output/'warm-start/dataset.json'
    if data_path.exists() and read(data_path)!=data:raise ValueError('frozen SFT data differs')
    atomic_json(data_path,data);order=sft_order(cfg);root=output/'warm-start';identity=digest({'config':cfg['sft'],'data':data,'order':order})
    completed=[]
    for p in sorted((root/'checkpoints').glob('step-*')):
        meta=read(p/'complete.json');verify_files(p,meta['files'])
        if meta['identity']!=identity or meta['step']!=len(completed)+1:raise ValueError('SFT checkpoint chain differs')
        completed.append(p)
    if len(completed)>steps:raise ValueError('SFT step cap exceeded')
    if (root/'B').exists():
        audit_sft(output,local=False);return
    torch.manual_seed(cfg['sft']['seed']);torch.cuda.manual_seed_all(cfg['sft']['seed']);random.seed(cfg['sft']['seed'])
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=under(ROOT,branch['base_dir']);tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,trust_remote_code=False)
    samples=[encode_example(tok,e,branch['sampling']['max_model_len']) for e in data]
    initial=completed[-1]/'adapter' if completed else child/'initial/adapter'
    base=AutoModelForCausalLM.from_pretrained(folder,local_files_only=True,trust_remote_code=False,
                                            torch_dtype=torch.float32,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,initial,adapter_name='policy',is_trainable=True).eval()
    for name,param in model.named_parameters():param.requires_grad_('lora_' in name and '.policy.' in name)
    configure_training_memory(model,branch);params=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in params)!=1089536:raise ValueError('SFT LoRA size differs')
    optimizer=torch.optim.AdamW(params,lr=cfg['sft']['learning_rate'],weight_decay=0)
    if completed:
        if file_hash(completed[-1]/'state.pt')!=read(completed[-1]/'complete.json')['state_sha256']:
            raise ValueError('SFT resume state hash differs')
        state=torch.load(completed[-1]/'state.pt',map_location='cpu',weights_only=False)
        if state['identity']!=identity or state['step']!=len(completed):raise ValueError('SFT resume identity differs')
        optimizer.load_state_dict(state['optimizer']);restore_rng(torch,state)
    with tqdm(total=exposures,initial=len(completed)*4,desc='Warm-start response CE',unit='action',file=sys.stdout,disable=False) as bar:
        for i in range(len(completed),steps):
            metrics=sft_batch(model,optimizer,[samples[j] for j in order[i]])
            if metrics['optimizer_steps']!=[i+1]:raise ValueError('SFT Adam counter differs')
            bar.set_postfix(step=i+1,loss=sum(metrics['losses'])/4,phase='backward',refresh=True);bar.update(4)
            temp=root/'checkpoints'/('.step-'+str(i+1)+'-'+str(time.time_ns()));temp.mkdir(parents=True)
            sha=save_policy_adapter(model,temp/'adapter')
            state={**rng_state(torch),'optimizer':optimizer.state_dict(),'identity':identity,'step':i+1}
            torch.save(state,temp/'state.pt')
            reloaded=torch.load(temp/'state.pt',map_location='cpu',weights_only=False)
            if reloaded['step']!=i+1 or not torch.equal(reloaded['torch_rng'],state['torch_rng']):raise ValueError('SFT state roundtrip differs')
            atomic_json(temp/'metrics.json',{'step':i+1,'indices':order[i],'adapter_sha256':sha,**metrics})
            atomic_json(temp/'complete.json',{'step':i+1,'identity':identity,'state_sha256':file_hash(temp/'state.pt'),
                                             'files':checkpoint_binding(temp)})
            target=root/'checkpoints'/f'step-{i+1:04}';temp.rename(target);completed.append(target)
    staging=root/('.B-'+str(time.time_ns()));shutil.copytree(completed[-1]/'adapter',staging/'adapter')
    shutil.copyfile(child/'initial/model_identity.json',staging/'model_identity.json')
    atomic_json(staging/'source_config.json',cfg['sft'])
    atomic_json(staging/'source_audit.json',{'passed':True,'identity':identity,'train_only':True,'actual_updates':steps,'action_exposures':exposures})
    atomic_json(staging/'complete.json',{'adapter_sha256':{p.name:file_hash(p) for p in (staging/'adapter').iterdir() if p.is_file()},
                'total_sft_updates':513+steps,'total_dpo_updates':11,'total_rl_updates':0,'parent_W_sha256':branch['adapter_sha256'],
                'sft_identity':identity,'source_checkpoint':f'checkpoints/step-{steps:04}','source_adapter_sha256':file_hash(completed[-1]/'adapter/adapter_model.safetensors')})
    staging.rename(root/'B');audit_sft(output,local=False)


def audit_sft(output,local=False):
    from safetensors.torch import load_file
    from grpo_worker import tensor_digest
    cfg=read(output/'config.json');data=read(output/'warm-start/dataset.json');verify_for(cfg,data)
    steps=cfg['sft']['steps'];exposures=steps*cfg['sft']['batch_size']
    if data!=supervision_for(cfg) or read(ROOT/cfg['sft_examples'])!=data:raise ValueError('SFT frozen examples differ')
    identity=digest({'config':cfg['sft'],'data':data,'order':sft_order(cfg)})
    root=output/'warm-start';paths=sorted((root/'checkpoints').glob('step-*'))
    if len(paths)!=steps:raise ValueError('all frozen SFT steps required')
    previous=load_file(str(output/'branches/w/initial/adapter/adapter_model.safetensors'))
    from tqdm import tqdm
    for i,p in enumerate(tqdm(paths,desc='Audit SFT checkpoints',unit='step',file=sys.stdout,disable=False)):
        meta=read(p/'complete.json');verify_files(p,meta['files']);metrics=read(p/'metrics.json')
        if (meta['identity']!=identity or meta['step']!=i+1 or metrics['indices']!=sft_order(cfg)[i]
                or metrics['optimizer_steps']!=[i+1] or metrics['frozen_parameter_gradients']!=0
                or not all(math.isfinite(x) for x in metrics['losses']) or not math.isfinite(metrics['gradient_norm'])):
            raise ValueError('SFT step metrics differ')
        if not local:
            import torch
            if file_hash(p/'state.pt')!=meta['state_sha256']:raise ValueError('SFT state hash differs')
            saved=torch.load(p/'state.pt',map_location='cpu',weights_only=False)
            if saved['identity']!=identity or saved['step']!=i+1 or sorted({int(v['step']) for v in saved['optimizer']['state'].values()})!=[i+1]:
                raise ValueError('SFT saved Adam identity differs')
        current=load_file(str(p/'adapter/adapter_model.safetensors'))
        if (metrics['before_tensor_sha256']!=tensor_digest(previous)
                or metrics['after_tensor_sha256']!=tensor_digest(current)
                or metrics['adapter_sha256']!=file_hash(p/'adapter/adapter_model.safetensors')):
            raise ValueError('SFT tensor/metric binding differs')
        if current.keys()!=previous.keys() or len(current)!=112 or not any(not current[k].equal(previous[k]) for k in current):
            raise ValueError('SFT actual LoRA update missing')
        previous=current
    complete=read(root/'B/complete.json');verify_files(root/'B/adapter',complete['adapter_sha256'])
    if (complete['total_sft_updates']!=513+steps or complete['total_dpo_updates']!=11
            or complete.get('total_rl_updates')!=0 or complete.get('sft_identity')!=identity
            or complete.get('parent_W_sha256')!=read(output/'branches/w/config.json')['adapter_sha256']
            or complete.get('source_checkpoint')!=f'checkpoints/step-{steps:04}'
            or read(root/'B/source_config.json')!=cfg['sft']
            or read(root/'B/source_audit.json')!={'passed':True,'identity':identity,'train_only':True,'actual_updates':steps,'action_exposures':exposures}
            or file_hash(root/'B/adapter/adapter_config.json')!=file_hash(paths[-1]/'adapter/adapter_config.json')
            or complete['source_adapter_sha256']!=file_hash(paths[-1]/'adapter/adapter_model.safetensors')
            or file_hash(root/'B/adapter/adapter_model.safetensors')!=complete['source_adapter_sha256']):
        raise ValueError('B export differs from final frozen step')
    result={'passed':True,'sft_updates':steps,'action_exposures':exposures,'identity':identity,'scope':'local_returned_files' if local else 'cloud_with_state'}
    atomic_json(root/('local_audit.json' if local else 'audit.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','diagnose','warm-start','audit-sft']);p.add_argument('--output',type=Path,required=True);p.add_argument('--name');p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.stage=='prepare':prepare_child(a.output,a.name)
    elif a.stage=='diagnose':diagnose(a.output)
    elif a.stage=='warm-start':warm_start(a.output)
    else:audit_sft(a.output,a.local)
