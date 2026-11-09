"""One bounded evaluation-only C4 job with per-episode resume and failure ZIP."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from grpo_c4_common import ROOT,read,atomic_json,verify_files,validate_config,LABELS
from run_v2_grpo import source_files,resource_snapshot,package_review
from streamed_process import run_logged


def resume_spent(prior,status=None,now=None):
    spent=max(prior.get('elapsed_seconds',0),(status or {}).get('elapsed_seconds',0))
    active=prior.get('active_started_unix')
    if active is not None:spent+=max(0,(time.time() if now is None else now)-active)
    return spent


def run(config,resume=False):
    from tqdm.auto import tqdm
    cfg=validate_config(read(config));release=read(ROOT/cfg.get('release_manifest','grpo_c4_release_manifest.json'));verify_files(ROOT,release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:raise ValueError('config must belong to published C4 release')
    output=ROOT/'artifacts'/cfg['run_id']
    if resume:
        if read(output/'config.json')!=cfg:raise ValueError('same C4 config required on resume')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);prior=read(output/'runner_state.json')
        spent=resume_spent(prior,read(output/'run_status.json') if (output/'run_status.json').exists() else None);attempt=prior['attempt']+1
    else:
        if output.exists():raise FileExistsError('C4 run exists; use --resume')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg);sources=source_files();atomic_json(output/'run_manifest.json',{'source_sha256':sources})
        for name in sources:
            dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        spent=0.;attempt=1
    archive=ROOT/(cfg['run_id']+('-review.zip' if attempt==1 else f'-review-attempt-{attempt:03}.zip'))
    if archive.exists():raise FileExistsError('C4 attempt archive already exists')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    started=time.monotonic();stages=[];error=None;stop='failed';active=None
    bar=tqdm(total=5,desc='C4 independent confirmation',unit='stage',file=sys.stdout,disable=False,dynamic_ncols=True)
    def elapsed():return spent+time.monotonic()-started
    def save():atomic_json(output/'runner_state.json',{'elapsed_seconds':elapsed(),'attempt':attempt,'active_started_unix':active,'new_training_updates':0})
    def stage(name,command,label=None):
        nonlocal active
        remaining=cfg['max_wall_seconds']-cfg['packaging_reserve_seconds']-elapsed()
        if remaining<=0:raise TimeoutError('C4 cumulative wall-time budget reached')
        active=time.time();save();log=output/f'{name}-attempt-{attempt:03}.log'
        print(f'\nStarting {name}; remaining {remaining:.0f}s; log: {log}',flush=True);resource_snapshot(output,name+'-before');begin=time.monotonic()
        args=[sys.executable,'-u',str(ROOT/'scripts/grpo_c4_worker.py'),command,'--output',str(output)]
        if label:args+=['--label',label]
        code=run_logged(args,cwd=ROOT,env=env,log_path=log,timeout=remaining)
        stages.append({'stage':name,'exit_code':code,'elapsed_seconds':time.monotonic()-begin});active=None;save();resource_snapshot(output,name+'-after')
        if code:raise RuntimeError(name+' failed; inspect '+log.name)
        bar.set_postfix(stage=name,refresh=True);bar.update(1)
    try:
        save()
        if not (output/'input_binding.json').exists():stage('prepare','prepare')
        else:bar.update(1)
        for label in LABELS:stage('evaluate-'+label,'evaluate',label)
        stage('audit','audit');stop='completed';verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}';stop='timeout' if isinstance(exc,(TimeoutError,subprocess.TimeoutExpired)) else 'failed'
    finally:
        active=None;save();bar.close();decision=read(output/'c4_audit.json').get('acceptance') if (output/'c4_audit.json').exists() else None
        status={'run_id':cfg['run_id'],'phase':'C4','attempt':attempt,'error':error,'stop_reason':stop,'elapsed_seconds':elapsed(),
            'max_wall_seconds':cfg['max_wall_seconds'],'new_training_updates':0,'stages':stages,'acceptance':decision,'archive':archive.name,
            'committed_episodes':{label:len(list((output/'episodes'/label).glob('*.json'))) for label in LABELS}}
        atomic_json(output/'run_status.json',status);atomic_json(output/'attempts'/f'attempt-{attempt:03}.json',status);package_review(output,archive)
        print('\nDownload: '+str(archive),flush=True);print('Run status: '+str({k:status[k] for k in ('error','stop_reason','elapsed_seconds','committed_episodes','acceptance')}),flush=True)
    if error:raise RuntimeError(error)
    return archive

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-c4-confirmation-v1.json');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
