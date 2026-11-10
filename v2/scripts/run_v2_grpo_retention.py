"""Bounded retention SFT then gated GRPO; both attempts share the original six-hour budget."""
import argparse,copy,json,shutil,threading,time
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,runtime_identity,initial_policy,validate_config
from run_v2_grpo import source_files
from run_v2_grpo_repair import Study as RepairStudy,BudgetStop,states,evaluation,select,resume_spent,package
from apt_v2.grpo_repair_study import b_gate
from apt_v2.grpo_retention_study import validate_master,validate_prior

MANIFEST='grpo_sql_retention_release_manifest.json'


def branch_config(cfg,output,name):
    c=copy.deepcopy(read(ROOT/cfg['branch_template']));c['run_id']=cfg['run_id']+'-'+name
    c['repair_arm']='warm' if name=='c' else 'plain'
    if name=='c':
        b=output/'warm-start/B';c['initial_dir']=b.relative_to(ROOT).as_posix()
        c['adapter_sha256']=file_hash(b/'adapter/adapter_model.safetensors');c['source_sft_updates']=528
    validate_config(c);return c


class Study(RepairStudy):
    def child(self,name):
        config=branch_config(self.cfg,self.output,name);p=self.output/'branch-configs'/(name+'.json')
        if p.exists() and read(p)!=config:raise ValueError('saved retention child config differs')
        atomic_json(p,config)
        self.stage('prepare-'+name,'grpo_repair_worker.py',['prepare','--output',str(self.output),'--name',name])
        return self.output/'branches'/name

    def execute(self):
        prior=read(ROOT/self.cfg['prior_evidence']);validate_prior(prior,self.cfg)
        atomic_json(self.output/'prior-evidence.json',prior)
        child=self.child('w');w=self.evaluate(child,'w','weighted_sft',0,initial_policy(read(child/'config.json')))
        if not w['evaluation_quality_passed']:
            return {'route':'warm_start','stop_reason':'baseline_quality_rejected','selected_candidate':None}
        # Reserve B evaluation and one final C evaluation before starting the one fixed SFT recipe.
        reserve=self.training_reserve(w['elapsed_seconds'],2)
        if self.cfg['max_wall_seconds']-self.elapsed()<reserve+180:
            raise BudgetStop('insufficient remaining combined budget for B and C evaluations')
        self.stage('warm-start','grpo_repair_worker.py',['warm-start','--output',str(self.output)],reserve)
        child=self.child('c');b=self.evaluate(child,'b','weighted_sft',0,initial_policy(read(child/'config.json')))
        gate=b_gate(w,b);atomic_json(self.output/'warm-start/b-gate.json',gate)
        if not gate['passed']:
            return {'route':'warm_start','stop_reason':'b_gate_rejected','selected_candidate':None,'b_gate':gate}
        results,complete=self.train(['c'],w)
        return {'route':'warm_start','stop_reason':'completed' if complete else 'completed_budget_stop',
                **select(w,results,b,complete)}


def run(config,resume=False):
    started=time.monotonic();cfg=read(config);validate_master(cfg);validate_prior(read(ROOT/cfg['prior_evidence']),cfg)
    release=read(ROOT/MANIFEST);verify_files(ROOT,release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:raise ValueError('config not published')
    output=ROOT/'artifacts'/cfg['run_id'];actual=runtime_identity()
    if actual['sqlite']!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    if resume:
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=actual:
            raise ValueError('resume config/hardware differs')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        if file_hash(ROOT/MANIFEST)!=read(output/'run_manifest.json')['release_sha256']:
            raise ValueError('resume publication identity differs')
        previous=read(output/'runner_state.json');spent=resume_spent(output);attempt=previous['attempt']+1
        if (output/'selection.json').exists() and previous.get('finished') and read(output/'run_status.json').get('error') is None:
            raise ValueError('completed retention study must not be restarted')
        atomic_json(output/'attempts'/f"attempt-{previous['attempt']:03}-runner-snapshot.json",previous)
    else:
        if output.exists():raise FileExistsError('run exists; resume matching unfinished run')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg);atomic_json(output/'execution_environment.json',actual)
        shutil.copyfile(ROOT/MANIFEST,output/'release_manifest.json')
        sources=source_files();atomic_json(output/'run_manifest.json',{'source_sha256':sources,'release_sha256':file_hash(ROOT/MANIFEST)})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        spent,attempt=0.,1
    study=Study(cfg,output,spent,attempt,started);study.save()
    thread=threading.Thread(target=study.heartbeat,daemon=True);thread.start();error=None;selection={}
    archive=ROOT/(cfg['run_id']+f'-review-attempt-{attempt:03}.zip')
    try:
        selection=study.execute();atomic_json(output/'selection_proposal.json',selection);atomic_json(output/'selection.json',selection)
        study.stage('root-audit','audit_v2_grpo_retention.py',['--output',str(output)])
    except BudgetStop as exc:
        selection={'selected_candidate':None,'stop_reason':'budget_stop','reason':str(exc)}
        atomic_json(output/'selection.json',selection)
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}';selection={'selected_candidate':None,'stop_reason':'failed'}
        atomic_json(output/'selection.json',selection)
    finally:
        study.stop_event.set();thread.join(timeout=10);study.save(finished=True);counts={};counter_errors={}
        for child in (output/'branches').glob('*'):
            if child.is_dir() and not child.name.startswith('.') and child.name!='w':
                try:
                    cursor,state=states(child);counts[child.name]={'groups':cursor,'updates':state['updates']}
                except Exception as exc:counter_errors[child.name]=str(exc)
        if counter_errors:error=error or 'committed counter verification failed';selection['selected_candidate']=None
        status={'actual_updates':{n:v['updates'] for n,v in counts.items()},'committed_groups':{n:v['groups'] for n,v in counts.items()},
                'sft_updates':len(list((output/'warm-start/checkpoints').glob('step-*'))),'counter_verification_errors':counter_errors,
                'run_id':cfg['run_id'],'attempt':attempt,'error':error,'max_wall_seconds':cfg['max_wall_seconds'],
                'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],'total_authorized_seconds':cfg['total_authorized_seconds'],
                'stages':study.stages,**selection}
        final=package(output,archive,status,study)
        receipt=read(archive.with_suffix('.receipt.json'))
        receipt.update(prior_elapsed_seconds=cfg['prior_elapsed_seconds'],
                       combined_elapsed_seconds=cfg['prior_elapsed_seconds']+receipt['elapsed_seconds_including_packaging'],
                       total_authorized_seconds=cfg['total_authorized_seconds'])
        receipt['combined_budget_passed']=receipt['combined_elapsed_seconds']<=cfg['total_authorized_seconds']
        atomic_json(archive.with_suffix('.receipt.json'),receipt);atomic_json(output/'packaging_receipt.json',receipt)
    if error:raise RuntimeError(error)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-sql-retention-study-v1.json')
    p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
