"""Finish E2 audit from saved artifacts; preserve original failed run and archive."""
import os,sys,time,shutil,zipfile
from pathlib import Path
from common import ROOT,read,write,file_hash,verify_files
from streamed_process import run_logged

RUN='dpo-uniform-control-v1-20261104'
MANIFEST='dpo_uniform_audit_repair_release_manifest.json'

def run():
    output=ROOT/'artifacts'/RUN;archive=ROOT/(RUN+'-audit-repaired-review.zip')
    if archive.exists() or (output/'audit_repair_status.json').exists():raise FileExistsError('audit repair output already exists')
    release=read(ROOT/MANIFEST);verify_files(ROOT,release['files'])
    original_archive=ROOT/(RUN+'-review.zip')
    if original_archive.exists() and file_hash(original_archive)!=release['original_review_archive_sha256']:raise ValueError('reviewed original archive differs')
    verify_files(output,release['reviewed_artifacts_sha256'])
    sources=read(output/'run_manifest.json')['source_sha256'];verify_files(ROOT,sources);verify_files(output/'source',sources)
    if sources!=release['original_source_sha256']:raise ValueError('reviewed original source differs')
    status=read(output/'run_status.json');train=read(output/'training_summary.json')
    if train['optimizer_updates']!=258 or status['optimizer_updates']!=258 or status['stages'][-1]!={'stage':'audit','exit_code':1} or any(s['exit_code'] for s in status['stages'][:-1]):raise ValueError('completed E2 training/evaluation and original audit failure required')
    if any((output/n).exists() for n in ('audit.json','results.csv','reference_numerics_audit.json')):raise FileExistsError('audit products already present')
    original={p.relative_to(output).as_posix():file_hash(p) for p in sorted(output.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    repair=output/'audit_repair_source';repair.mkdir()
    for name in [*release['files'],MANIFEST]:
        p=repair/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    write(output/'audit_repair_input_binding.json',{'original_artifacts_sha256':original,'original_run_status_sha256':file_hash(output/'run_status.json'),'original_review_archive_sha256':release['original_review_archive_sha256'],'original_archive_verified':original_archive.exists(),'repair_files_sha256':release['files']})
    started=time.monotonic();error=None;code=None
    try:
        code=run_logged([sys.executable,'-u',str(ROOT/'scripts/audit_v2_dpo_uniform_repair.py'),'--output',str(output)],cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8'),log_path=output/'audit-repair.log',timeout=1800)
        if code:raise RuntimeError('saved-artifact audit failed; inspect audit-repair.log')
        verify_files(output,original);verify_files(ROOT,release['files'])
        if not read(output/'audit.json')['passed'] or not read(output/'reference_numerics_audit.json')['passed']:raise ValueError('complete repaired audit required')
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}'
    finally:
        write(output/'audit_repair_status.json',{'elapsed_seconds':time.monotonic()-started,'exit_code':code,'error':error,'training_executed':False,'inference_executed':False,'optimizer_updates_this_repair':0,'original_training_updates':258,'original_run_status_preserved':file_hash(output/'run_status.json')==original['run_status.json'],'original_artifacts_preserved':all(file_hash(output/n)==h for n,h in original.items())})
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(output.rglob('*')):
                rel=p.relative_to(output).as_posix()
                if p.is_file() and p.suffix not in ('.pt','.pyc') and (p.suffix!='.safetensors' or rel=='final/adapter/adapter_model.safetensors'):z.write(p,p.relative_to(output.parent).as_posix())
        print('Download: '+str(archive),flush=True)
    if error:raise RuntimeError(error)

if __name__=='__main__':run()
