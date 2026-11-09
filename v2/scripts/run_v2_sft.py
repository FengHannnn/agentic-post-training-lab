"""Fresh bounded cloud pilot; preserve evidence and review ZIP on every exit."""
import os,sys,time,signal,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files

def run():
    cfg=read(ROOT/'configs/sft-pilot-v1.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    release=ROOT/'sft_release_manifest.json';verify_files(ROOT,read(release)['files'])
    model=ROOT/'.cache/qwen2.5-1.5b-instruct/model_identity.json'
    if not model.exists():raise FileNotFoundError('expected cached model: '+str(model))
    output.mkdir(parents=True);write(output/'config.json',cfg)
    sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn')
    try:
        for stage,script in [('train','train_v2_sft.py'),('evaluate','evaluate_v2_sft.py')]:
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('SFT job budget reached')
            with (output/(stage+'.log')).open('x',encoding='utf8') as log:
                proc=subprocess.Popen([sys.executable,str(ROOT/'scripts'/script),'--output',str(output)],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt')
                try:code=proc.wait(timeout=remaining)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                    else:proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                        else:proc.kill()
                        proc.wait()
                    raise TimeoutError('SFT stage stopped at budget: '+stage)
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
                if path.is_file():z.write(path,path.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
