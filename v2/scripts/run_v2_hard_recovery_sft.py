"""Weighted hard-recovery SFT with live progress, epoch probes and a review archive."""
import os,sys,time,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged

def run():
    cfg=read(ROOT/'configs/hard-recovery-sft-weighted-v1.json');run_id=cfg['run_id'];output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    verify_files(ROOT,read(ROOT/'hard_recovery_sft_release_manifest.json')['files'])
    required={cfg['initial_run']:('model_identity.json','audit.json','final/complete.json','final/adapter/adapter_model.safetensors','after_chosen_scores.json'),
        cfg['data_run']:('audit.json','pairs.jsonl'),cfg['diagnostic_run']:('audit.json','config.json','input_binding.json')}
    for source_run,names in required.items():
        for name in names:
            p=ROOT/'artifacts'/source_run/name
            if not p.is_file():raise FileNotFoundError('weighted SFT prerequisite missing: '+str(p))
    output.mkdir(parents=True);write(output/'config.json',cfg);sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    started=time.monotonic();stages=[];error=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    try:
        for stage,script,args in [('train','train_v2_hard_recovery_sft.py',[]),('checkpoints','evaluate_v2_hard_recovery_sft.py',['--mode','checkpoints']),
            ('recovery','evaluate_v2_hard_recovery_sft.py',['--mode','recovery']),('greedy','evaluate_v2_hard_recovery_sft.py',['--mode','greedy']),
            ('sample','evaluate_v2_hard_recovery_sft.py',['--mode','sample']),('audit','audit_v2_hard_recovery_sft.py',[])]:
            remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
            if remaining<=0:raise TimeoutError('weighted hard-recovery job budget reached')
            print('Starting '+stage+'; log: '+str(output/(stage+'.log')),flush=True)
            try:
                code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),'--output',str(output),*args],
                    cwd=ROOT,env=env,log_path=output/(stage+'.log'),timeout=remaining)
            except subprocess.TimeoutExpired:raise TimeoutError('weighted recovery stage stopped at budget: '+stage)
            stages.append({'stage':stage,'exit_code':code});print(stage+': '+str(code),flush=True)
            if code:raise RuntimeError(stage+' failed; inspect '+str(output/(stage+'.log')))
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,
            'max_wall_seconds':cfg['max_wall_seconds'],'cost_estimation_enabled':False})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                rel=p.relative_to(output).as_posix()
                if p.is_file() and p.suffix not in ('.pt','.pyc') and (p.suffix!='.safetensors' or rel=='final/adapter/adapter_model.safetensors'):
                    z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
