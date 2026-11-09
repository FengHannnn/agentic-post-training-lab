"""Run finite SFT diagnostics on an existing pilot, with no training updates."""
import os,sys,time,signal,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,file_hash,source_files,verify_files

def run():
    cfg=read(ROOT/'configs/sft-diagnostics-v1.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh diagnostic output required: '+run_id)
    verify_files(ROOT,read(ROOT/'sft_diagnostics_release_manifest.json')['files'])
    pilot=ROOT/'artifacts'/cfg['pilot_run']
    for name in ('examples.jsonl','data_summary.json','model_identity.json','run_manifest.json','final/complete.json'):
        if not (pilot/name).is_file():raise FileNotFoundError('pilot evidence missing: '+str(pilot/name))
    complete=read(pilot/'final/complete.json');verify_files(pilot/'final/adapter',complete['adapter_sha256'])
    expected='95f908b1e9163907f9e72eb2c5aa5442da873193f7fb730352ed35382c7b379e'
    if file_hash(pilot/'final/adapter/adapter_model.safetensors')!=expected:raise ValueError('expected reviewed 63-step adapter')
    output.mkdir(parents=True);write(output/'config.json',cfg)
    sources=source_files();sources['diagnostics/diagnose_sft_v1.py']=file_hash(ROOT/'diagnostics/diagnose_sft_v1.py')
    write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1')
    try:
        for stage in ('prepare','hf-fp32','hf-bf16','vllm','audit'):
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('diagnostic time budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            with (output/(stage+'.log')).open('x',encoding='utf8') as log:
                proc=subprocess.Popen([sys.executable,str(ROOT/'diagnostics/diagnose_sft_v1.py'),
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
                if p.is_file():z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
