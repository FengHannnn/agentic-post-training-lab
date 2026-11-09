"""Paired frozen SFT/DPO independent evaluation with bounded execution."""
import os,sys,time,signal,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,file_hash,source_files,verify_files

def run():
    cfg=read(ROOT/'configs/dpo-independent-test-v1.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh diagnostic output required: '+run_id)
    verify_files(ROOT,read(ROOT/'dpo_test_release_manifest.json')['files'])
    candidate=ROOT/'artifacts'/cfg['candidate_run']
    for name in ('pairs.jsonl','model_identity.json','final/complete.json','final/adapter/adapter_model.safetensors'):
        if not (candidate/name).is_file():raise FileNotFoundError('candidate file missing: '+str(candidate/name))
    output.mkdir(parents=True);write(output/'config.json',cfg)
    sources=source_files()
    write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    try:
        for stage in ('prepare','greedy','sample','audit'):
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('diagnostic time budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            with (output/(stage+'.log')).open('x',encoding='utf8') as log:
                proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/test_v2_dpo.py'),
                    '--stage',stage,'--output',str(output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    start_new_session=os.name!='nt')
                try:code=proc.wait(timeout=remaining)
                except (subprocess.TimeoutExpired,KeyboardInterrupt):
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                    else:proc.terminate()
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                        else:proc.kill()
                        proc.wait()
                    raise TimeoutError('diagnostic stage stopped: '+stage)
            stages.append({'stage':stage,'exit_code':code})
            print(stage+': '+str(code),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,
            'error':error,'training_enabled':False,'optimizer_updates':0,'max_wall_seconds':cfg['max_wall_seconds']})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix not in ('.safetensors','.pt'):
                    z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
