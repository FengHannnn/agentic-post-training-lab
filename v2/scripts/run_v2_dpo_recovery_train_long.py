"""Three-epoch recovery DPO with live terminal output and a three-hour budget."""
import os,sys,time,signal,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged

def run():
    cfg=read(ROOT/'configs/dpo-recovery-train-v2.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    release=ROOT/'dpo_recovery_train_v2_release_manifest.json';verify_files(ROOT,read(release)['files'])
    model=ROOT/'.cache/qwen2.5-1.5b-instruct/model_identity.json'
    if not model.exists():raise FileNotFoundError('expected cached model: '+str(model))
    parent=ROOT/'artifacts'/cfg['initial_run']
    for name in ('model_identity.json','final/adapter/adapter_model.safetensors'):
        if not (parent/name).is_file():raise FileNotFoundError('initial DPO file missing: '+str(parent/name))
    data=ROOT/'artifacts'/cfg['data_run']/'pairs.jsonl'
    if not data.is_file():raise FileNotFoundError('DPO pairs missing: '+str(data))
    output.mkdir(parents=True);write(output/'config.json',cfg)
    sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    try:
        for stage,script,args in [('train','train_v2_dpo_recovery_long.py',[]),('greedy','evaluate_v2_dpo_recovery.py',['--mode','greedy']),
                ('sample','evaluate_v2_dpo_recovery.py',['--mode','sample']),('recovery','evaluate_v2_dpo_recovery.py',['--mode','recovery']),('audit','audit_v2_dpo_recovery.py',[])]:
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('recovery DPO job budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            try:
                code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),'--output',str(output),*args],
                    cwd=ROOT,env=env,log_path=output/(stage+'.log'),timeout=remaining)
            except subprocess.TimeoutExpired:raise TimeoutError('recovery DPO stage stopped at budget: '+stage)
            stages.append({'stage':stage,'exit_code':code})
            print(stage+': '+str(code)+'; log: '+str(output/(stage+'.log')),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,
            'max_wall_seconds':cfg['max_wall_seconds'],'cost_estimation_enabled':False})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(output.rglob('*')):
                if path.is_file() and path.suffix!='.pt' and (path.suffix!='.safetensors' or path.relative_to(output).as_posix()=='final/adapter/adapter_model.safetensors'):
                    z.write(path,path.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
