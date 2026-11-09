"""Inference-only train/development recovery diagnostic with live progress."""
import os,sys,time,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged

def run():
    cfg=read(ROOT/'configs/recovery-context-diagnostics-v1.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    verify_files(ROOT,read(ROOT/'recovery_context_diagnostic_release_manifest.json')['files'])
    for spec in cfg['checkpoints'].values():
        folder=ROOT/'artifacts'/spec['run']
        for name in ('audit.json','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
            if not (folder/name).is_file():raise FileNotFoundError('diagnostic checkpoint missing: '+str(folder/name))
    if not (ROOT/'artifacts'/cfg['data_run']/'pairs.jsonl').is_file():raise FileNotFoundError('reviewed preference pairs missing')
    output.mkdir(parents=True);write(output/'config.json',cfg);sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    try:
        for stage in ('prepare','evaluate','audit'):
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('recovery diagnostic job budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            try:
                code=run_logged([sys.executable,'-u',str(ROOT/'scripts/diagnose_v2_recovery_contexts.py'),'--output',str(output),'--mode',stage],
                    cwd=ROOT,env=env,log_path=output/(stage+'.log'),timeout=remaining)
            except subprocess.TimeoutExpired:raise TimeoutError('recovery diagnostic stage stopped at budget: '+stage)
            stages.append({'stage':stage,'exit_code':code});print(stage+': '+str(code),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,
            'max_wall_seconds':cfg['max_wall_seconds'],'training_updates':0,'cost_estimation_enabled':False})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix not in ('.pt','.safetensors','.pyc'):z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
