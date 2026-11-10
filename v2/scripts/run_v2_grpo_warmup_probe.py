"""Thirty-minute train-only diagnosis of W, B24 and B15; no optimizer updates."""
import argparse,shutil,threading,time
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,runtime_identity
from run_v2_grpo import source_files
from run_v2_grpo_repair import Study,BudgetStop,resume_spent,package
from apt_v2.grpo_warmup_probe import validate_config,validate_prior

MANIFEST='grpo_warmup_probe_release_manifest.json'


def run(config,resume=False):
    started=time.monotonic();cfg=read(config);validate_config(cfg)
    validate_prior(read(ROOT/cfg['input_dir']/'prior-evidence.json'),cfg)
    release=read(ROOT/MANIFEST);verify_files(ROOT,release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:raise ValueError('probe config not published')
    output=ROOT/'artifacts'/cfg['run_id'];actual=runtime_identity()
    if actual['sqlite']!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    if resume:
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=actual:raise ValueError('resume config/hardware differs')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        if file_hash(ROOT/MANIFEST)!=read(output/'run_manifest.json')['release_sha256']:raise ValueError('resume release differs')
        previous=read(output/'runner_state.json');spent=resume_spent(output);attempt=previous['attempt']+1
        if previous.get('finished') and read(output/'run_status.json').get('error') is None:
            raise ValueError('completed probe must not be restarted')
        atomic_json(output/'attempts'/f"attempt-{previous['attempt']:03}-runner-snapshot.json",previous)
    else:
        if output.exists():raise FileExistsError('probe exists; resume matching unfinished attempt')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg);atomic_json(output/'execution_environment.json',actual)
        shutil.copyfile(ROOT/MANIFEST,output/'release_manifest.json');sources=source_files()
        atomic_json(output/'run_manifest.json',{'source_sha256':sources,'release_sha256':file_hash(ROOT/MANIFEST)})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        spent,attempt=0.,1
    study=Study(cfg,output,spent,attempt,started);study.save();thread=threading.Thread(target=study.heartbeat,daemon=True);thread.start()
    error=None;reason='diagnostic_completed';archive=ROOT/(cfg['run_id']+f'-review-attempt-{attempt:03}.zip')
    try:
        study.stage('prepare','grpo_warmup_probe_worker.py',['prepare','--output',str(output)],120)
        for name in cfg['models']:
            study.stage('probe-'+name,'grpo_warmup_probe_worker.py',['model','--output',str(output),'--model',name],120)
        study.stage('audit','grpo_warmup_probe_worker.py',['audit','--output',str(output)],60)
    except BudgetStop as exc:error=str(exc);reason='budget_stop'
    except (Exception,KeyboardInterrupt) as exc:error=f'{type(exc).__name__}: {exc}';reason='failed'
    finally:
        study.stop_event.set();thread.join(timeout=10);study.save(finished=True)
        status={'run_id':cfg['run_id'],'attempt':attempt,'error':error,'stop_reason':reason,'selected_candidate':None,
                'max_wall_seconds':cfg['max_wall_seconds'],'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],
                'total_authorized_seconds':cfg['total_authorized_seconds'],'stages':study.stages,
                'actual_updates':{},'sft_updates':0,'models_completed':[name for name in cfg['models'] if (output/'probes'/name/'complete.json').exists()],
                'committed_contexts':{name:len(list((output/'probes'/name/'rows').glob('*.json'))) for name in cfg['models']}}
        final=package(output,archive,status,study);receipt=read(archive.with_suffix('.receipt.json'))
        receipt.update(prior_elapsed_seconds=cfg['prior_elapsed_seconds'],combined_elapsed_seconds=cfg['prior_elapsed_seconds']+receipt['elapsed_seconds_including_packaging'],
                       total_authorized_seconds=cfg['total_authorized_seconds'])
        receipt['combined_budget_passed']=receipt['combined_elapsed_seconds']<=cfg['total_authorized_seconds']
        atomic_json(archive.with_suffix('.receipt.json'),receipt);atomic_json(output/'packaging_receipt.json',receipt)
    if error:raise RuntimeError(error)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-sql-warmup-probe-v1.json')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
