"""E4 expert continuation CE with live progress, paired evaluation and failure archive."""
import os,sys,time,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged

def resource_snapshot(output,label):
    # Read-only host/GPU context; collection failure never changes the experiment.
    commands=[['nvidia-smi','--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory','--format=csv'],['free','-h'],['df','-h',str(ROOT)],['ps','-eo','pid,ppid,pcpu,pmem,rss,stat,comm','--sort=-rss']]
    import json
    with (output/'resource_snapshots.jsonl').open('a',encoding='utf8') as stream:
        for cmd in commands:
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,timeout=2)
                row={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'label':label,'command':cmd,'exit_code':r.returncode,'stdout':r.stdout[:16000],'stderr':r.stderr[:2000]}
            except Exception as exc:row={'timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'label':label,'command':cmd,'error':str(exc)}
            stream.write(json.dumps(row)+'\n')

def run():
    cfg=read(ROOT/'configs/dpo-inference-confirmation-v1.json');run_id=cfg['run_id'];output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    verify_files(ROOT,read(ROOT/'dpo_inference_confirmation_release_manifest.json')['files'])
    output.mkdir(parents=True);write(output/'config.json',cfg);sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    try:
        for stage,script,args in [('prepare','dpo_inference_confirmation_common.py',[]),*[(b,'evaluate_v2_dpo_inference_confirmation.py',['--branch',b]) for b in cfg['evaluation_branches']],('audit','audit_v2_dpo_inference_confirmation.py',[])]:
            resource_snapshot(output,stage+'-before')
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('Inference confirmation job budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            try:
                code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),'--output',str(output),*args],
                    cwd=ROOT,env=env,log_path=output/(stage+'.log'),timeout=remaining)
            except subprocess.TimeoutExpired:raise TimeoutError('Inference confirmation stage stopped at budget: '+stage)
            resource_snapshot(output,stage+'-after')
            stages.append({'stage':stage,'exit_code':code});print(stage+': '+str(code),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,
            'max_wall_seconds':cfg['max_wall_seconds'],'cost_estimation_enabled':False,'training_enabled':False,'optimizer_updates':0})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                rel=p.relative_to(output).as_posix()
                if p.is_file() and p.suffix not in ('.pt','.pyc') and p.suffix!='.safetensors':
                    z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
