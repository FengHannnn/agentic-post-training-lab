"""One cumulative-budget job: W baseline, matched outcome branches and full audit."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,verify_files
from grpo_curriculum_common import matched_configs,states_at,resume_spent,BRANCHES
from run_v2_grpo import source_files,resource_snapshot,package_review
from streamed_process import run_logged


def run(config,resume=False):
    from tqdm.auto import tqdm
    cfg,branches=matched_configs(config)
    release=read(ROOT/'grpo_curriculum_release_manifest.json');verify_files(ROOT,release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:
        raise ValueError('config must belong to published curriculum release')
    output=ROOT/'artifacts'/cfg['run_id']
    if resume:
        if read(output/'config.json')!=cfg:raise ValueError('identical curriculum config required for resume')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        prior=read(output/'runner_state.json')
        spent=resume_spent(prior,read(output/'run_status.json') if (output/'run_status.json').exists() else None)
        attempt=prior['attempt']+1
    else:
        if output.exists():raise FileExistsError('curriculum run exists; use --resume')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg)
        sources=source_files();atomic_json(output/'run_manifest.json',{'source_sha256':sources})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        for b,c in branches.items():
            child=output/'branches'/b;atomic_json(child/'config.json',c)
            atomic_json(child/'run_manifest.json',{'source_sha256':sources});shutil.copytree(output/'source',child/'source')
        spent=0.;attempt=1
    started=time.monotonic();stages=[];error=None;stop='failed';active=None
    children={b:output/'branches'/b for b in BRANCHES};states={b:states_at(child) for b,child in children.items()}
    refreshed={b:read(child/'runner_state.json').get('refresh_only_last_group',False) if (child/'runner_state.json').exists() else False for b,child in children.items()}
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    archive=ROOT/(cfg['run_id']+('-review.zip' if attempt==1 else f'-review-attempt-{attempt:03}.zip'))
    if archive.exists():raise FileExistsError('attempt archive already exists')
    count=cfg['groups_per_branch'];label=f'grpo_g{count:04}'
    bar=tqdm(total=4*count+8,desc='Outcome curriculum',unit='stage',file=sys.stdout,disable=False,dynamic_ncols=True)
    def elapsed():return spent+time.monotonic()-started
    def save():
        value=elapsed()
        atomic_json(output/'runner_state.json',{'elapsed_seconds':value,'attempt':attempt,'active_started_unix':active,
            'policies':{b:s[-1] for b,s in states.items()},'next_groups':{b:len(s)-1 for b,s in states.items()}})
        for b,child in children.items():
            atomic_json(child/'runner_state.json',{'policy':states[b][-1],'next_group':len(states[b])-1,
                'elapsed_seconds':value,'attempt':attempt,'refresh_only_last_group':refreshed[b]})
    def progress(name):
        completed=sum((c/'input_binding.json').exists() for c in children.values())
        completed+=(children['control']/'evaluations/weighted_sft.json').exists()
        for child in children.values():
            completed+=len(list((child/'transitions').glob('group-*.json')))
            completed+=len(list((child/'groups').glob('group-*.json')))
            completed+=(child/'evaluations'/(label+'.json')).exists()
        completed+=(output/'curriculum_audit.json').exists()
        bar.n=completed;bar.set_postfix(stage=name,control_updates=states['control'][-1]['updates'],
                                      lr10x_updates=states['lr10x'][-1]['updates'],refresh=True)
    def stage(name,script,args,target):
        nonlocal active
        remaining=cfg['max_wall_seconds']-cfg['packaging_reserve_seconds']-elapsed()
        if remaining<=0:raise TimeoutError('cumulative curriculum wall-time budget reached')
        active=time.time();save();log=output/f'{name}-attempt-{attempt:03}.log'
        print(f'\nStarting {name}; remaining {remaining:.0f}s; log: {log}',flush=True)
        resource_snapshot(output,name+'-before');begin=time.monotonic()
        code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),*args,'--output',str(target)],
                        cwd=ROOT,env=env,log_path=log,timeout=remaining)
        stages.append({'stage':name,'exit_code':code,'elapsed_seconds':time.monotonic()-begin})
        active=None;save();resource_snapshot(output,name+'-after')
        if code:raise RuntimeError(name+' failed; inspect '+log.name)
        progress(name)
    def evaluate(b,model_label,policy,cursor):
        child=children[b];spec={'label':model_label,'group_cursor':cursor,'policy':policy}
        path=child/'evaluations'/(model_label+'-policy.json')
        if path.exists() and read(path)!=spec:raise ValueError('frozen evaluation spec differs')
        atomic_json(path,spec)
        stage('evaluate-'+b+'-'+model_label,'evaluate_v2_grpo.py',['--label',model_label,'--spec',str(path)],child)
    try:
        save();progress('resume' if resume else 'start')
        for b,child in children.items():
            if not (child/'input_binding.json').exists():stage('prepare-'+b,'grpo_worker.py',['prepare'],child)
        identities=[read(c/'execution_environment.json') for c in children.values()]
        if identities[0]!=identities[1]:raise ValueError('same GPU/software required for both branches')
        if (output/'execution_environment.json').exists() and read(output/'execution_environment.json')!=identities[0]:
            raise ValueError('resume GPU/software identity differs')
        atomic_json(output/'execution_environment.json',identities[0])
        p=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
        if p.returncode:raise ValueError('GPU driver capture failed')
        driver={'stdout':p.stdout,'stderr':p.stderr,'exit_code':p.returncode}
        if (output/'driver_identity.json').exists() and read(output/'driver_identity.json')!=driver:
            raise ValueError('within-job GPU driver identity differs')
        atomic_json(output/'driver_identity.json',driver)
        evaluate('control','weighted_sft',states['control'][0],0)
        for i in range(count):
            for b,child in children.items():
                if len(states[b])-1>i:continue
                policy=child/'policies'/f'before-group-{i:04}.json';atomic_json(policy,states[b][-1])
                args=['--group-index',str(i),'--policy-file',str(policy)]
                stage(f'collect-{b}-{i:04}','grpo_worker.py',['collect',*args],child)
                stage(f'update-{b}-{i:04}','grpo_worker.py',['update',*args],child)
                states[b]=states_at(child);save();progress('committed-'+b)
        for b,child in children.items():
            evaluate(b,label,states[b][-1],count)
            policy=child/'policies'/f'before-group-{count:04}.json';atomic_json(policy,states[b][-1])
            refreshed[b]=True;save()
            stage('refresh-'+b,'grpo_worker.py',['collect','--group-index',str(count),'--policy-file',str(policy)],child)
        stage('audit','audit_v2_grpo_curriculum.py',[],output);stop='group_cap'
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}'
        stop='timeout' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'failed'
    finally:
        active=None;states={b:states_at(child) for b,child in children.items()};save();bar.close()
        status={'run_id':cfg['run_id'],'phase':'outcome_curriculum','attempt':attempt,'error':error,'stop_reason':stop,
            'elapsed_seconds':elapsed(),'max_wall_seconds':cfg['max_wall_seconds'],
            'actual_updates':{b:s[-1]['updates'] for b,s in states.items()},
            'committed_groups':{b:len(s)-1 for b,s in states.items()},'stages':stages,'archive':archive.name,
            'state_pt_kept_on_cloud':True}
        atomic_json(output/'run_status.json',status);atomic_json(output/'attempts'/f'attempt-{attempt:03}.json',status)
        package_review(output,archive)
        print('\nDownload: '+str(archive),flush=True)
        print('Run status: '+str({k:status[k] for k in ('error','stop_reason','actual_updates','committed_groups')}),flush=True)
    if error:raise RuntimeError(error)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-outcome-curriculum-v1.json')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)

