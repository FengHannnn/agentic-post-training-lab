"""One bounded C3 job: matched outcome/fine branches on a newly bound L40S."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,initial_policy
from grpo_c3_common import matched_configs,states_at
from run_v2_grpo import source_files,resource_snapshot,package_review
from streamed_process import run_logged


def run(config,resume=False):
    from tqdm.auto import tqdm
    cfg,branches=matched_configs(config)
    release=read(ROOT/'grpo_c3_release_manifest.json');verify_files(ROOT,release['files'])
    output=ROOT/'artifacts'/cfg['run_id']
    if resume:
        if read(output/'config.json')!=cfg:raise ValueError('identical matched config required for resume')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        prior=read(output/'runner_state.json');spent=max(prior['elapsed_seconds'],read(output/'run_status.json')['elapsed_seconds'] if (output/'run_status.json').exists() else 0);attempt=prior['attempt']+1
    else:
        if output.exists():raise FileExistsError('matched run exists; use --resume')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg)
        sources=source_files();atomic_json(output/'run_manifest.json',{'source_sha256':sources})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        for b,c in branches.items():
            child=output/'branches'/b;atomic_json(child/'config.json',c);atomic_json(child/'run_manifest.json',{'source_sha256':sources})
            shutil.copytree(output/'source',child/'source')
        spent=0.;attempt=1
    started=time.monotonic();stages=[];error=None;stop='failed'
    children={b:output/'branches'/b for b in branches};states={b:states_at(child) for b,child in children.items()}
    refreshed={b:read(child/'runner_state.json').get('refresh_only_last_group',False) if (child/'runner_state.json').exists() else False for b,child in children.items()}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    archive=ROOT/(cfg['run_id']+('-review.zip' if attempt==1 else f'-review-attempt-{attempt:03}.zip'))
    if archive.exists():raise FileExistsError('attempt archive already exists')
    bar=tqdm(total=104,desc='C3 matched',unit='stage',file=sys.stdout,disable=False,dynamic_ncols=True)
    def save():
        elapsed=spent+time.monotonic()-started
        atomic_json(output/'runner_state.json',{'elapsed_seconds':elapsed,'attempt':attempt,'policies':{b:s[-1] for b,s in states.items()},'next_groups':{b:len(s)-1 for b,s in states.items()}})
        for b,child in children.items():atomic_json(child/'runner_state.json',{'policy':states[b][-1],'next_group':len(states[b])-1,'elapsed_seconds':elapsed,'attempt':attempt,'refresh_only_last_group':refreshed[b]})
    def stage(name,script,args,target):
        save();remaining=cfg['max_wall_seconds']-cfg['packaging_reserve_seconds']-(spent+time.monotonic()-started)
        if remaining<=0:raise TimeoutError('C3 matched wall-time budget reached')
        log=output/f'{name}-attempt-{attempt:03}.log';print(f'\nStarting {name}; remaining {remaining:.0f}s; log: {log}',flush=True)
        resource_snapshot(output,name+'-before');begin=time.monotonic()
        code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),*args,'--output',str(target)],cwd=ROOT,env=env,log_path=log,timeout=remaining)
        stages.append({'stage':name,'exit_code':code,'elapsed_seconds':time.monotonic()-begin});save();resource_snapshot(output,name+'-after')
        if code:raise RuntimeError(name+' failed; inspect '+log.name)
        bar.set_postfix(stage=name,outcome_updates=states['outcome'][-1]['updates'],fine_updates=states['fine'][-1]['updates'],refresh=True);bar.update(1)
    def evaluate(b,label,policy,cursor):
        child=children[b];spec={'label':label,'group_cursor':cursor,'policy':policy};path=child/'evaluations'/(label+'-policy.json')
        if path.exists() and read(path)!=spec:raise ValueError('frozen matched evaluation spec differs')
        atomic_json(path,spec);stage('evaluate-'+b+'-'+label,'evaluate_v2_grpo.py',['--label',label,'--spec',str(path)],child)
    try:
        for b,child in children.items():
            if not (child/'input_binding.json').exists():stage('prepare-'+b,'grpo_worker.py',['prepare'],child)
        identities=[read(c/'execution_environment.json') for c in children.values()]
        if identities[0]!=identities[1]:raise ValueError('same new GPU/software identity required for both branches')
        if (output/'execution_environment.json').exists() and read(output/'execution_environment.json')!=identities[0]:raise ValueError('matched resume GPU identity differs')
        atomic_json(output/'execution_environment.json',identities[0])
        if not (output/'driver_identity.json').exists():
            p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
            if p.returncode:raise ValueError('GPU driver identity capture failed')
            atomic_json(output/'driver_identity.json',{'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode})
        evaluate('outcome','weighted_sft',states['outcome'][0],0)
        for i in range(cfg['groups_per_branch']):
            for b,child in children.items():
                if len(states[b])-1>i:continue
                policy=child/'policies'/f'before-group-{i:04}.json';atomic_json(policy,states[b][-1])
                args=['--group-index',str(i),'--policy-file',str(policy)]
                stage(f'collect-{b}-{i:04}','grpo_worker.py',['collect',*args],child)
                stage(f'update-{b}-{i:04}','grpo_worker.py',['update',*args],child)
                states[b]=states_at(child);save()
        for b,child in children.items():
            evaluate(b,'grpo_g0024',states[b][-1],24)
            policy=child/'policies/before-group-0024.json';atomic_json(policy,states[b][-1]);refreshed[b]=True;save()
            stage('refresh-'+b,'grpo_worker.py',['collect','--group-index','24','--policy-file',str(policy)],child)
        stage('audit','audit_v2_grpo_c3.py',[],output);stop='group_cap'
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}';stop='timeout' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'failed'
    finally:
        states={b:states_at(child) for b,child in children.items()};save();bar.close()
        status={'run_id':cfg['run_id'],'phase':'C3','attempt':attempt,'error':error,'stop_reason':stop,'elapsed_seconds':spent+time.monotonic()-started,'max_wall_seconds':cfg['max_wall_seconds'],'actual_updates':{b:s[-1]['updates'] for b,s in states.items()},'committed_groups':{b:len(s)-1 for b,s in states.items()},'stages':stages,'archive':archive.name,'state_pt_kept_on_cloud':True}
        atomic_json(output/'run_status.json',status);atomic_json(output/'attempts'/f'attempt-{attempt:03}.json',status);package_review(output,archive)
        print('\nDownload: '+str(archive),flush=True);print('Run status: '+str({k:status[k] for k in ('error','stop_reason','actual_updates','committed_groups')}),flush=True)
    if error:raise RuntimeError(error)
    return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-c3-matched-v1.json');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
