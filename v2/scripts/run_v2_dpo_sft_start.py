"""One bounded DPO trajectory; evaluate each candidate before allowing continuation."""
import os,sys,time,subprocess,zipfile,shutil
from pathlib import Path
from common import ROOT,read,write,source_files,verify_files
from streamed_process import run_logged
from run_v2_dpo_early86 import resource_snapshot

def update_count(output):
    import json
    values=[0]
    for p in (output/'training').glob('segment-*-steps.jsonl'):
        for line in p.read_text(encoding='utf8').splitlines():
            try:values.append(json.loads(line)['step'])
            except (ValueError,KeyError):pass
    return max(values)

def run():
    cfg=read(ROOT/'configs/dpo-sft-start-short-v1.json');run_id=cfg['run_id'];output=ROOT/'artifacts'/run_id;archive=ROOT/(run_id+'-review.zip')
    if output.exists() or archive.exists():raise FileExistsError('fresh run ID required: '+run_id)
    verify_files(ROOT,read(ROOT/'dpo_sft_start_release_manifest.json')['files'])
    output.mkdir(parents=True);write(output/'config.json',cfg);sources=source_files();write(output/'run_manifest.json',{'source_sha256':sources})
    for name in sources:
        p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    started=time.monotonic();stages=[];error=None;stop_reason=None
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
    def stage(name,script,args=None):
        resource_snapshot(output,name+'-before');remaining=cfg['max_wall_seconds']-30-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('Short DPO job budget reached')
        print('Starting '+name+'; log: '+str(output/(name+'.log')),flush=True)
        try:code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),'--output',str(output),*(args or [])],cwd=ROOT,env=env,log_path=output/(name+'.log'),timeout=remaining)
        except subprocess.TimeoutExpired:raise TimeoutError('Short DPO stage stopped at budget: '+name)
        stages.append({'stage':name,'exit_code':code});resource_snapshot(output,name+'-after');print(name+': '+str(code),flush=True)
        if code:raise RuntimeError(name+' failed; inspect '+str(output/(name+'.log')))
    try:
        stage('prepare','dpo_sft_start_common.py')
        stage('weighted_sft','evaluate_v2_dpo_sft_start.py',['--branch','weighted_sft']);stage('audit-weighted_sft','audit_v2_dpo_sft_start.py',['--branch','weighted_sft'])
        if read(output/'weighted_sft-decision.json')['training_allowed']:
            for step in cfg['checkpoint_steps']:
                b=f'dpo_{step:04}';stage('train-'+b,'train_v2_dpo_sft_start.py',['--target-step',str(step)]);stage(b,'evaluate_v2_dpo_sft_start.py',['--branch',b]);stage('audit-'+b,'audit_v2_dpo_sft_start.py',['--branch',b]);decision=read(output/(b+'-decision.json'))
                if decision['passed'] or decision['severe_regression_stop']:
                    stop_reason='first_passing_checkpoint' if decision['passed'] else 'severe_regression';break
            else:stop_reason='step_cap_reached'
        else:stop_reason='baseline_changed'
        stage('audit','audit_v2_dpo_sft_start.py')
        if read(output/'selection.json')['stop_reason']!=stop_reason:raise ValueError('runner stop reason differs from audited selection')
        verify_files(ROOT,sources)
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'run_status.json',{'elapsed_seconds':time.monotonic()-started,'stages':stages,'error':error,'stop_reason':stop_reason,'max_wall_seconds':cfg['max_wall_seconds'],'training_enabled':True,'optimizer_updates':update_count(output),'maximum_optimizer_updates':cfg['max_steps'],'cost_estimation_enabled':False})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                if p.is_file() and p.suffix not in ('.pt','.pyc'):z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':run()
