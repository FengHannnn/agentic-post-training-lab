"""Evaluate three frozen checkpoints on the independent fine GRPO follow-up."""
import argparse
import shutil
import subprocess
import threading
import time
from pathlib import Path

from grpo_common import ROOT, atomic_json, file_hash, read, runtime_identity, verify_files
from apt_v2 import grpo_fine_closeout_followup as spec
from apt_v2.grpo_evaluation import evaluation_key, paired_evaluation
from apt_v2.grpo_fine_closeout import selection
from apt_v2.grpo_lr_midpoint import completion_diagnostics
from run_v2_grpo import source_files
from run_v2_grpo_fine_closeout import persist
from run_v2_grpo_repair import BudgetStop, Study, evaluation, package, resume_spent

RECOVERY_SLICES = ('sql_unverified_requery','stale_evidence_repair','failed_check_repair')


def branch_config(name):
    template=read(ROOT/spec.TEMPLATE)
    cfg={**template,'run_id':f'{spec.RUN_ID}-{name}',
         'credit_method':spec.CLOSEOUT if name=='closeout' else spec.LOCAL}
    spec.validate_branch(cfg)
    return cfg


def label(name):
    return 'weighted_sft' if name=='w' else 'grpo_g0048'


def bind_checkpoint(child, name, prepare=False):
    policy=spec.POLICIES[name]
    folder=child/policy['checkpoint']/'adapter'
    if name!='w':
        frozen=ROOT/spec.INPUT/'models'/name/'adapter'
        expected={p.relative_to(frozen).as_posix():file_hash(p) for p in frozen.rglob('*') if p.is_file()}
        if prepare and not folder.exists():
            staging=folder.with_name('.adapter-'+str(time.time_ns()))
            shutil.copytree(frozen,staging)
            staging.rename(folder)
        verify_files(folder,expected)
    if file_hash(folder/'adapter_model.safetensors')!=policy['adapter_sha256']:
        raise ValueError('frozen evaluation checkpoint differs')
    return policy


def milestones(episode):
    rows=[p['row'] for p in episode['start']['prefix']]+episode['turns']
    found={metric for row in rows for metric in row['new_verified_metrics']}
    return {'both_monthly':{'current_net_cents','prior_net_cents'}<=found,
            'all_three':{'current_net_cents','prior_net_cents','change_cents'}<=found,
            'checked':any(row['action'].get('tool')=='check_result' and row['observation'].get('passed') is True
                          for row in episode['turns']),
            'submitted':episode['success']}


def stage_pairs(a,b,cfg):
    left,right=({evaluation_key(e):e for e in result['episodes']} for result in (a,b))
    if len(left)!=396 or len(right)!=396 or left.keys()!=right.keys() or any(
        (e['start_id'],e['seed'],e['sampling'])!=(right[k]['start_id'],right[k]['seed'],right[k]['sampling'])
        for k,e in left.items()
    ):
        raise ValueError('independent evaluation pair differs')
    rows={k:(milestones(left[k]),milestones(right[k])) for k in left}
    stages={}
    for mode in ('sample','greedy'):
        for kind in ('complete','recovery'):
            pairs=[x for k,x in rows.items() if k[2]==mode and bool(k[1])==(kind=='recovery')]
            shared=[x for x in pairs if all(item['all_three'] for item in x)]
            stages[f'{mode}_{kind}']={'episodes':len(pairs),
                'before':{k:sum(x[0][k] for x in pairs) for k in ('both_monthly','all_three','checked','submitted')},
                'after':{k:sum(x[1][k] for x in pairs) for k in ('both_monthly','all_three','checked','submitted')},
                'both_certified':len(shared),
                'shared_certified_submit_gained':sum(not x[0]['submitted'] and x[1]['submitted'] for x in shared),
                'shared_certified_submit_lost':sum(x[0]['submitted'] and not x[1]['submitted'] for x in shared)}
    slices={}
    for scenario in RECOVERY_SLICES:
        pairs=[x for k,x in rows.items() if k[2]=='sample' and k[1]==scenario]
        slices[scenario]={'episodes':len(pairs),'before':sum(x[0]['submitted'] for x in pairs),
                          'after':sum(x[1]['submitted'] for x in pairs),
                          'gained':sum(not x[0]['submitted'] and x[1]['submitted'] for x in pairs),
                          'lost':sum(x[0]['submitted'] and not x[1]['submitted'] for x in pairs)}
    return {'standard':paired_evaluation(a['episodes'],b['episodes'],cfg),'stages':stages,'recovery_slices':slices}


def result_record(results):
    w=results['w']
    branches={name:results[name] for name in spec.NAMES if name!='w'}
    for ev in branches.values():
        ev['effective_bf16_change']=(ev['worker_proof'][0]['loaded_tensor_sha256']
                                     !=w['worker_proof'][0]['loaded_tensor_sha256'])
    comparisons={f'{a}_to_{b}':stage_pairs(results[a],results[b],branch_config('w'))
                 for a,b in (('w','local'),('w','closeout'),('local','closeout'))}
    return {**selection(w,branches,minimum_credit_gain=spec.GAIN),'paired':comparisons,
            'targeted':{name:result['targeted_summary'] for name,result in results.items()},
            'evaluated_policies':spec.POLICIES,'new_training_updates':0,'stop_reason':'completed'}


class Followup(Study):
    def execute(self):
        results={}
        for name in spec.NAMES:
            cfg=branch_config(name)
            persist(self.output/'branch-configs'/(name+'.json'),cfg)
            self.stage('prepare-'+name,'grpo_repair_worker.py',['prepare','--output',str(self.output),'--name',name])
            child=self.output/'branches'/name
            if read(child/'execution_environment.json')!=read(self.output/'execution_environment.json'):
                raise ValueError('same-job runtime differs')
            policy=bind_checkpoint(child,name,prepare=True)
            results[name]=self.evaluate(child,name,label(name),0 if name=='w' else 48,policy)
        return result_record(results)

    def stage(self,name,script,args,tail_reserve=180.,stage_timeout=None):
        if name.startswith('evaluate-'):
            tail_reserve=max(tail_reserve,480.)
        return super().stage(name,script,args,tail_reserve,stage_timeout)


def audit(output,local=False):
    cfg=read(output/'config.json')
    spec.validate_master(cfg)
    spec.validate_evidence(read(ROOT/spec.INPUT/'source-evidence.json'))
    source=read(output/'run_manifest.json')
    verify_files(ROOT,source['source_sha256'])
    verify_files(output/'source',source['source_sha256'])
    if file_hash(output/'release_manifest.json')!=source['release_sha256']:
        raise ValueError('returned release snapshot differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files'])
    results={}
    for name in spec.NAMES:
        child=output/'branches'/name
        if (read(child/'config.json')!=branch_config(name)
                or read(child/'execution_environment.json')!=read(output/'execution_environment.json')
                or (child/'transitions').exists()):
            raise ValueError('frozen evaluation branch differs')
        policy=bind_checkpoint(child,name)
        ev=evaluation(child,label(name),local)
        if ev['spec']!={'label':label(name),'group_cursor':0 if name=='w' else 48,'policy':policy}:
            raise ValueError('recorded evaluation policy differs')
        results[name]=ev
    expected=result_record(results)
    if read(output/'selection.json')!=expected:
        raise ValueError('independent pairing or decision differs')
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_evaluation',
            'new_training_updates':0,'selection':expected}
    atomic_json(output/('local_audit.json' if local else 'audit.json'),result)
    return result


def run(config,resume=False):
    started=time.monotonic()
    cfg=read(config)
    spec.validate_master(cfg)
    release=ROOT/spec.MANIFEST
    verify_files(ROOT,read(release)['files'])
    if config.resolve().relative_to(ROOT).as_posix()!=spec.CONFIG:
        raise ValueError('run the frozen follow-up config')
    spec.validate_evidence(read(ROOT/spec.INPUT/'source-evidence.json'))
    output=ROOT/'artifacts'/spec.RUN_ID
    runtime=runtime_identity()
    if runtime['sqlite']!='3.54.1':
        raise ValueError('SQLite 3.54.1 required')
    if resume:
        previous=read(output/'runner_state.json')
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=runtime:
            raise ValueError('resume config or hardware differs')
        manifest=read(output/'run_manifest.json')
        verify_files(ROOT,manifest['source_sha256'])
        if manifest['release_sha256']!=file_hash(release):
            raise ValueError('resume release differs')
        if previous.get('finished') and read(output/'run_status.json').get('error') is None:
            raise ValueError('completed follow-up must not restart')
        spent,attempt=resume_spent(output),previous['attempt']+1
        persist(output/'attempts'/f"attempt-{previous['attempt']:03}-runner-snapshot.json",previous)
    else:
        output.mkdir(parents=True)
        atomic_json(output/'config.json',cfg)
        atomic_json(output/'execution_environment.json',runtime)
        shutil.copyfile(release,output/'release_manifest.json')
        sources=source_files()
        atomic_json(output/'run_manifest.json',{'source_sha256':sources,'release_sha256':file_hash(release)})
        for rel in sources:
            dst=output/'source'/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/rel,dst)
        spent,attempt=0.,1
    job=Followup(cfg,output,spent,attempt,started)
    job.save()
    heartbeat=threading.Thread(target=job.heartbeat,daemon=True)
    heartbeat.start()
    error=None
    chosen={}
    archive=ROOT/(spec.RUN_ID+f'-review-attempt-{attempt:03}.zip')
    try:
        chosen=job.execute()
        atomic_json(output/'selection.json',chosen)
        job.stage('root-audit','run_v2_grpo_fine_closeout_followup.py',['--audit',str(output)])
    except (BudgetStop,subprocess.TimeoutExpired) as exc:
        chosen={'selected_candidate':None,'formal_baseline':'W','stop_reason':'budget_stop','reason':str(exc)}
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}'
        chosen={'selected_candidate':None,'formal_baseline':'W','stop_reason':'failed'}
    finally:
        job.stop_event.set()
        heartbeat.join(timeout=10)
        job.save(finished=True)
        status={'run_id':spec.RUN_ID,'attempt':attempt,'error':error,'actual_updates':{},
                'sft_updates':0,'new_training_updates':0,'max_wall_seconds':spec.CAP,'stages':job.stages,**chosen}
        package(output,archive,status,job)
    if error:
        raise RuntimeError(error)
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/spec.CONFIG)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--local',action='store_true')
    args=parser.parse_args()
    if args.audit:
        audit(args.audit,args.local)
    else:
        run(args.config,args.resume)
