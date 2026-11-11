"""Ordered warm-start search; select the first B passing train and full gates."""
import argparse,copy,shutil,threading,time
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,runtime_identity,initial_policy,validate_config
from run_v2_grpo import source_files
from run_v2_grpo_repair import Study as RepairStudy,BudgetStop,evaluation,resume_spent,package
from apt_v2.grpo_dose_study import MANIFEST
from grpo_dose_worker import protocol_for, ready_record, save_completion_diagnostics
from apt_v2.grpo_repair_study import b_gate, learning_rate_protocol

def branch_config(cfg,output,name):
    c=copy.deepcopy(read(ROOT/cfg['branch_template']));c['run_id']=cfg['run_id']+'-'+name
    c['repair_arm']='plain' if name=='w' else 'warm'
    if name!='w':
        protocol = protocol_for(cfg)
        recipe = protocol.recipe(protocol.CANDIDATES[name])
        folder = output/'warmups'/name/'warm-start/B'
        c['initial_dir'] = folder.relative_to(ROOT).as_posix()
        c['adapter_sha256'] = file_hash(folder/'adapter/adapter_model.safetensors')
        c['source_sft_updates'] = 513 + recipe['steps']
        c['warm_target_repeats'] = recipe['target_repeats']
        if learning_rate_protocol(c['repair_profile']) is not None:
            c['warm_learning_rate'] = recipe['learning_rate']
    validate_config(c);return c


class Study(RepairStudy):
    def child(self,name):
        c=branch_config(self.cfg,self.output,name);p=self.output/'branch-configs'/(name+'.json')
        if p.exists() and read(p)!=c:raise ValueError('dose child config differs')
        atomic_json(p,c);self.stage('prepare-'+name,'grpo_repair_worker.py',['prepare','--output',str(self.output),'--name',name])
        return self.output/'branches'/name

    def execute(self):
        protocol = protocol_for(self.cfg)
        protocol.validate_prior(read(ROOT/self.cfg['input_dir']/'prior-evidence.json'), self.cfg)
        child=self.child('w');self.stage('prepare-probes','grpo_dose_worker.py',['prepare','--output',str(self.output)])
        self.stage('probe-w','grpo_dose_worker.py',['probe','--output',str(self.output),'--name','w'])
        w=self.evaluate(child,'w','weighted_sft',0,initial_policy(read(child/'config.json')));records=[]
        b = None
        if not w['evaluation_quality_passed']:
            save_completion_diagnostics(protocol, self.output, w, b, write=True)
            return protocol.selection(w, records)
        reserve=max(2400.,1.25*w['elapsed_seconds'])+360.
        for name in protocol.CANDIDATES:
            self.stage('warm-'+name,'grpo_dose_worker.py',['warm','--output',str(self.output),'--name',name],reserve)
            self.stage('probe-'+name,'grpo_dose_worker.py',['probe','--output',str(self.output),'--name',name],reserve)
            self.stage('gate-'+name,'grpo_dose_worker.py',['gate','--output',str(self.output),'--name',name],reserve)
            train=read(self.output/'train-gates'/(name+'.json'));full=None
            if train['passed']:
                child=self.child(name);b=self.evaluate(child,name,'weighted_sft',0,initial_policy(read(child/'config.json')))
                full=b_gate(w,b);p=self.output/'full-gates'/(name+'.json')
                if p.exists() and read(p)!=full:raise ValueError('saved full gate differs')
                atomic_json(p,full)
            from grpo_dose_worker import candidate_source
            identity=candidate_source(self.output,name)
            record={**protocol.candidate_fields(name),'train_gate':train,'full_gate':full,'adapter_sha256':identity['adapter_sha256']};p=self.output/'decisions'/(name+'.json')
            if p.exists() and read(p)!=record:raise ValueError('saved candidate decision differs')
            atomic_json(p,record);records.append(record)
            if full and full['passed']:
                atomic_json(self.output/'ready-B.json', ready_record(self.cfg, name, identity))
                break
        save_completion_diagnostics(protocol, self.output, w, b, write=True)
        return protocol.selection(w,records)


def committed_sft_counts(output):
    counts={};errors={}
    for root in (output/'warmups').glob('*'):
        if not root.is_dir() or root.name.startswith('.'):continue
        count=0;identity=None
        try:
            for i,path in enumerate(sorted((root/'warm-start/checkpoints').glob('step-*'))):
                meta=read(path/'complete.json');verify_files(path,meta['files']);metrics=read(path/'metrics.json')
                if meta['step']!=i+1 or metrics['optimizer_steps']!=[i+1] or (identity is not None and meta['identity']!=identity):
                    raise ValueError('committed SFT counter chain differs')
                identity=meta['identity'];count+=1
        except Exception as exc:errors[root.name]=str(exc)
        counts[root.name]=count
    return counts,errors


def run(config,resume=False):
    started = time.monotonic()
    cfg = read(config)
    protocol = protocol_for(cfg)
    protocol.validate_master(cfg)
    protocol.validate_prior(read(ROOT/cfg['input_dir']/'prior-evidence.json'), cfg)
    manifest = ROOT/protocol.MANIFEST
    release = read(manifest)
    verify_files(ROOT, release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:raise ValueError('dose config not published')
    output=ROOT/'artifacts'/cfg['run_id'];actual=runtime_identity()
    if actual['sqlite']!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    if resume:
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=actual:raise ValueError('dose resume config/hardware differs')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        if file_hash(manifest)!=read(output/'run_manifest.json')['release_sha256']:raise ValueError('dose resume publication differs')
        previous=read(output/'runner_state.json');spent=resume_spent(output);attempt=previous['attempt']+1
        if previous.get('finished') and read(output/'run_status.json').get('error') is None:raise ValueError('completed dose study must not restart')
        atomic_json(output/'attempts'/f"attempt-{previous['attempt']:03}-runner-snapshot.json",previous)
    else:
        if output.exists():raise FileExistsError('dose run exists; resume matching unfinished attempt')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg);atomic_json(output/'execution_environment.json',actual)
        shutil.copyfile(manifest,output/'release_manifest.json');sources=source_files()
        atomic_json(output/'run_manifest.json',{'source_sha256':sources,'release_sha256':file_hash(manifest)})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        spent,attempt=0.,1
    study=Study(cfg,output,spent,attempt,started);study.save();thread=threading.Thread(target=study.heartbeat,daemon=True);thread.start()
    error=None;chosen={};archive=ROOT/(cfg['run_id']+f'-review-attempt-{attempt:03}.zip')
    try:
        chosen=study.execute();atomic_json(output/'selection.json',chosen)
        study.stage('root-audit','grpo_dose_worker.py',['audit','--output',str(output)])
    except BudgetStop as exc:
        chosen={'selected_warm_start':None,'selected_candidate':None,'formal_baseline':'W','stop_reason':'budget_stop','reason':str(exc)}
        atomic_json(output/'selection.json',chosen)
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}';chosen={'selected_warm_start':None,'selected_candidate':None,'formal_baseline':'W','stop_reason':'failed'}
        atomic_json(output/'selection.json',chosen)
    finally:
        study.stop_event.set();thread.join(timeout=10);study.save(finished=True)
        counts,counter_errors=committed_sft_counts(output)
        if counter_errors:
            error=error or 'committed SFT counter verification failed'
            chosen={'selected_warm_start':None,'selected_candidate':None,'formal_baseline':'W','stop_reason':'failed'}
            atomic_json(output/'selection.json',chosen)
        status={'run_id':cfg['run_id'],'attempt':attempt,'error':error,'max_wall_seconds':cfg['max_wall_seconds'],
                'new_authorized_seconds':cfg['new_authorized_seconds'],'prior_elapsed_seconds':cfg['prior_elapsed_seconds'],'total_authorized_seconds':cfg['total_authorized_seconds'],
                'counter_verification_errors':counter_errors,'actual_updates':{},'actual_sft_updates':counts,'sft_updates':sum(counts.values()),'actual_rl_updates':0,'stages':study.stages,**chosen}
        final=package(output,archive,status,study)
        receipt=read(archive.with_suffix('.receipt.json'));receipt.update(new_authorized_seconds=cfg['new_authorized_seconds'],prior_elapsed_seconds=cfg['prior_elapsed_seconds'],
            combined_elapsed_seconds=cfg['prior_elapsed_seconds']+receipt['elapsed_seconds_including_packaging'],total_authorized_seconds=cfg['total_authorized_seconds'])
        receipt['combined_budget_passed']=receipt['combined_elapsed_seconds']<=cfg['total_authorized_seconds']
        atomic_json(archive.with_suffix('.receipt.json'),receipt);atomic_json(output/'packaging_receipt.json',receipt)
    if error:raise RuntimeError(error)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-sql-dose-study-v1.json');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.config,a.resume)
