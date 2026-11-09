"""Run the inference-only independent hard-recovery test with live logs and packaging."""
import os,sys,time,subprocess,zipfile,shutil
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged

def run():
    cfg=read(ROOT/'configs/hard-recovery-independent-test-v1.json');run_id=cfg['run_id']
    output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh test run ID required: '+run_id)
    verify_files(ROOT,read(ROOT/'hard_recovery_test_release_manifest.json')['files'])
    for source in (cfg['parent_run'],cfg['candidate_run']):
        for name in ('final/complete.json','final/adapter/adapter_model.safetensors','model_identity.json','audit.json','examples.jsonl','pairs.jsonl'):
            p=ROOT/'artifacts'/source/name
            if not p.is_file():raise FileNotFoundError('independent test prerequisite missing: '+str(p))
    output.mkdir(parents=True);write(output/'config.json',cfg);sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    try:
        for stage in ('prepare','greedy','sample','recovery','audit'):
            script='audit_v2_hard_recovery_test.py' if stage=='audit' else 'test_v2_hard_recovery.py'
            args=[] if stage=='audit' else ['--stage',stage]
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('independent test wall budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            try:
                code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),'--output',str(output),*args],
                    cwd=ROOT,env=env,log_path=output/(stage+'.log'),timeout=remaining)
            except subprocess.TimeoutExpired:raise TimeoutError('independent test stage stopped at budget: '+stage)
            stages.append({'stage':stage,'exit_code':code});print(stage+': '+str(code),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,
            'max_wall_seconds':cfg['max_wall_seconds'],'training_enabled':False,'optimizer_updates':0,'cost_estimation_enabled':False})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix not in ('.pt','.pyc','.safetensors'):z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
