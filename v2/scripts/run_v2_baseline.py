"""One bounded cloud baseline, with explicit model download and evidence packaging."""
import sys,os,json,subprocess,time,signal,zipfile,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,OUTPUT,read,write,source_files,verify_files
from prepare_v2 import preflight

def run(download_model=False):
    cfg=read(ROOT/'configs/baseline.json')
    limit=cfg['max_wall_seconds']-10
    if limit<=0:raise ValueError('positive job time budget required')
    archive=ROOT/'v2-baseline-review-20261029.zip'
    if OUTPUT.exists() or archive.exists():raise FileExistsError('fresh output and review archive required')
    verify_files(ROOT,read(ROOT/'bundle_manifest.json')['files'])
    if not download_model and not (ROOT/'.cache/qwen2.5-1.5b-instruct/model_identity.json').exists():
        raise ValueError('model absent; use --download-model for the first cloud run')
    result=preflight()
    OUTPUT.mkdir(parents=True)
    write(OUTPUT/'preflight.json',result)
    sources=source_files();write(OUTPUT/'run_manifest.json',{'config':cfg,'source_sha256':sources})
    for name in sources:
        dest=OUTPUT/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    started=time.perf_counter();stages=[];error=None
    commands=[]
    if download_model:commands.append(('model',ROOT/'scripts/prepare_v2.py',['--download-model']))
    commands.extend([('evaluate',ROOT/'scripts/evaluate_v2.py',[]),('audit',ROOT/'scripts/audit_v2.py',[])])
    try:
        for stage,script,args in commands:
            remaining=limit-(time.perf_counter()-started)
            if remaining<=0:raise TimeoutError('baseline time budget exhausted')
            proc=subprocess.Popen([sys.executable,str(script),*args],cwd=ROOT,start_new_session=os.name!='nt')
            try:code=proc.wait(timeout=remaining)
            except (subprocess.TimeoutExpired,KeyboardInterrupt):
                if os.name!='nt':os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                    else:proc.kill()
                    proc.wait()
                raise TimeoutError('baseline stopped at its budget')
            stages.append({'stage':stage,'exit_code':code})
            if code:raise RuntimeError(stage+' failed; later stages stopped')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        elapsed=time.perf_counter()-started
        write(OUTPUT/'cost_and_run.json',{'elapsed_seconds':elapsed,'cost_estimation_enabled':False,'max_job_seconds':limit,
            'stages':stages,'error':error,'automatic_retry':False,'automatic_shutdown':False,
            'scope':'model preparation, evaluation and audit'})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(OUTPUT.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(OUTPUT.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--download-model',action='store_true')
    args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight()))
    else:run(args.download_model)
