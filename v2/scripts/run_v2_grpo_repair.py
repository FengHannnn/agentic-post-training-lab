"""Six-hour gated SQL-repair study; completed evidence is reused and audited on resume."""
import subprocess
import argparse,copy,json,os,shutil,sys,threading,time,zipfile
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,runtime_identity,initial_policy,load_transition
from run_v2_grpo import source_files,resource_snapshot
from streamed_process import run_logged
from apt_v2.grpo_repair_study import SCHEMA,b_gate,candidate_checks,standard_counts,target_counts

MANIFEST='grpo_sql_repair_release_manifest.json'


class BudgetStop(RuntimeError):pass


def branch_config(cfg,output,name):
    c=copy.deepcopy(read(ROOT/cfg['branch_template']))
    c['run_id']=cfg['run_id']+'-'+name
    c['repair_arm']='warm' if name=='c' else ('plain' if name=='w' else name)
    if name=='c':
        b=output/'warm-start/B';c['initial_dir']=b.relative_to(ROOT).as_posix()
        c['adapter_sha256']=file_hash(b/'adapter/adapter_model.safetensors');c['source_sft_updates']=537
    from grpo_common import validate_config
    validate_config(c);return c


def states(child):
    cfg=read(child/'config.json');state=initial_policy(cfg);cursor=0
    while (child/'transitions'/f'group-{cursor:04}.json').exists():
        state=load_transition(child,cursor,state)['after'];cursor+=1
    if cursor>cfg.get('max_groups',36):raise ValueError('training group cap exceeded')
    return cursor,state


def evaluation(child,label,local=False):
    from transformers import AutoTokenizer
    from evaluate_v2_grpo import verify_evaluation,verify_targeted_evaluation
    data=read(child/'evaluations'/(label+'.json'));spec=data['spec']
    tok=AutoTokenizer.from_pretrained(child/'tokenizer',local_files_only=True,trust_remote_code=False)
    a=verify_evaluation(child,label,spec,tok,local=local);b=verify_targeted_evaluation(child,label,spec,tok)
    status=read(child/'evaluations'/(label+'-quality-status.json'))
    if status['evaluation_sha256']!=file_hash(child/'evaluations'/(label+'.json')) or status['passed']!=(a['passed'] and b['passed']):
        raise ValueError('evaluation quality status differs')
    return {**data,'evaluation_quality_passed':status['passed']}


def select(w,results,b=None,complete=True):
    checks={name:candidate_checks(w,result,b) for name,result in results.items()}
    parent=b if b is not None else w
    for name,check in checks.items():
        policy=results[name]['spec']['policy'];initial=parent['spec']['policy']
        check['checks']['actual_rl_update']=policy['updates']>0 and policy['adapter_sha256']!=initial['adapter_sha256']
        check['checks']['effective_bf16_change']=bool(results[name].get('effective_bf16_change',False))
        check['passed']=all(check['checks'].values())
    good=[n for n,c in checks.items() if c['passed'] and complete]
    primary=1 if standard_counts(w)[0]==16 else 0
    def key(name):
        s=standard_counts(results[name]);t=target_counts(results[name])
        return (s[primary],t[0],s[1-primary],t[1],s[2],s[3],name=='control')
    return {'selected_candidate':max(good,key=key) if good else None,'checks':checks,'complete_training_recipe':complete,
            'primary_metric':'recovery' if primary else 'complete','all_final_prompts_without_reminder':True,
            'B_is_a_separate_SFT_result':b is not None}


class Study:
    def __init__(self,cfg,output,spent=0.,attempt=1,started=None):
        self.cfg,self.output,self.spent,self.attempt=cfg,output,spent,attempt
        self.started=time.monotonic() if started is None else started;self.stages=[];self.stop_event=threading.Event();self.active=None
        self.env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',
                      VLLM_NO_USAGE_STATS='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')

    def elapsed(self):return self.spent+time.monotonic()-self.started

    def save(self,finished=False):
        atomic_json(self.output/'runner_state.json',{'attempt':self.attempt,'elapsed_seconds':self.elapsed(),
                    'active_stage':self.active,'finished':finished,'stages':self.stages})

    def heartbeat(self):
        while not self.stop_event.wait(5):
            atomic_json(self.output/'heartbeat.json',{'attempt':self.attempt,'elapsed_seconds':self.elapsed()})

    def stage(self,name,script,args,tail_reserve=180.,stage_timeout=None):
        remaining=self.cfg['max_wall_seconds']-self.elapsed()-tail_reserve
        if stage_timeout is not None:
            remaining = min(remaining, stage_timeout)
        if remaining<=0:raise BudgetStop('reserved final work reached before '+name)
        self.active=name;self.save();start=time.monotonic()
        log=self.output/f'{name}-attempt-{self.attempt:03}.log'
        print(f'\n{name}: {remaining:.0f}s available; log {log.name}',flush=True)
        resource_snapshot(self.output,name+'-before')
        try:
            code=run_logged([sys.executable,'-u',str(ROOT/'scripts'/script),*args],cwd=ROOT,env=self.env,log_path=log,timeout=remaining)
        except subprocess.TimeoutExpired:
            self.stages.append({'stage':name,'exit_code':124,'elapsed_seconds':time.monotonic()-start})
            self.active=None;self.save()
            raise
        self.stages.append({'stage':name,'exit_code':code,'elapsed_seconds':time.monotonic()-start})
        self.active=None;self.save();resource_snapshot(self.output,name+'-after')
        if code:raise RuntimeError(name+' failed; inspect '+log.name)

    def child(self,name):
        config=branch_config(self.cfg,self.output,name);p=self.output/'branch-configs'/(name+'.json')
        if p.exists() and read(p)!=config:raise ValueError('saved child configuration differs')
        atomic_json(p,config)
        self.stage('prepare-'+name,'grpo_repair_worker.py',['prepare','--output',str(self.output),'--name',name])
        return self.output/'branches'/name

    def evaluate(self,child,name,label,cursor,state):
        spec={'label':label,'group_cursor':cursor,'policy':state};p=child/'evaluations'/(label+'-policy.json')
        if p.exists() and read(p)!=spec:raise ValueError('evaluation policy spec differs')
        atomic_json(p,spec)
        self.stage('evaluate-'+name,'evaluate_v2_grpo.py',['--output',str(child),'--label',label,'--spec',str(p)])
        return evaluation(child,label)

    def training_reserve(self,baseline_seconds,models):
        return models*max(2400.,1.25*baseline_seconds)+240.+180.+180.

    def train(self,names,w):
        children={n:self.child(n) for n in names}
        baseline_seconds=w['elapsed_seconds'];reserve=self.training_reserve(baseline_seconds,len(names))
        with tqdm(total=36*len(names),desc='Paired GRPO groups',unit='group',file=sys.stdout,disable=False) as bar:
            bar.update(sum(states(c)[0] for c in children.values()))
            for cycle in range(0,36,2):
                if all(states(c)[0]>=cycle+2 for c in children.values()):continue
                timings=[s['elapsed_seconds'] for s in self.stages if s['stage'].startswith(('collect-','update-'))]
                estimate=2*len(names)*max(150.,1.25*sum(timings)/max(1,len(timings)//2)) if timings else 360.*len(names)
                decision={'elapsed_seconds':self.elapsed(),'reserved_final_seconds':reserve,'next_cycle_seconds':estimate,
                          'continue':self.cfg['max_wall_seconds']-self.elapsed()>=reserve+estimate,'cycle':cycle}
                atomic_json(self.output/'budget-decisions'/f'{self.attempt:03}-{cycle:04}.json',decision)
                if not decision['continue']:break
                for index in (cycle,cycle+1):
                    for name,child in children.items():
                        cursor,state=states(child)
                        if cursor>index:continue
                        if cursor!=index:raise ValueError('paired training cursor differs')
                        p=child/'policies'/f'before-group-{index:04}.json';atomic_json(p,state)
                        args=['--output',str(child),'--group-index',str(index),'--policy-file',str(p)]
                        self.stage(f'collect-{name}-{index:04}','grpo_worker.py',['collect',*args],reserve)
                        self.stage(f'update-{name}-{index:04}','grpo_worker.py',['update',*args],reserve)
                        cursor,state=states(child);bar.update(1)
                        atomic_json(child/'runner_state.json',{'policy':state,'next_group':cursor,'refresh_only_last_group':False})
        cursors=[states(c)[0] for c in children.values()]
        if len(set(cursors))!=1 or cursors[0]%2:raise ValueError('unequal paired stop')
        if not cursors[0]:raise BudgetStop('no complete training cycle fits the reserved budget')
        results={}
        for name,child in children.items():
            cursor,state=states(child);p=child/'policies'/f'before-group-{cursor:04}.json';atomic_json(p,state)
            atomic_json(child/'runner_state.json',{'policy':state,'next_group':cursor,'refresh_only_last_group':True})
            self.stage('refresh-'+name,'grpo_worker.py',['collect','--output',str(child),'--group-index',str(cursor),'--policy-file',str(p)])
            if cursor:self.stage('audit-'+name,'audit_v2_grpo.py',['--output',str(child)])
            results[name]={**self.evaluate(child,name,f'grpo_g{cursor:04}',cursor,state),
                           'effective_bf16_change':bool(read(child/'audit.json').get('effective_bf16_tensor_hash_changed',False))}
        return results,cursors[0]==36

    def execute(self):
        wchild=self.child('w');w=self.evaluate(wchild,'w','weighted_sft',0,initial_policy(read(wchild/'config.json')))
        if not w['evaluation_quality_passed']:
            return {'stop_reason':'baseline_quality_rejected','selected_candidate':None}
        self.stage('diagnostic','grpo_repair_worker.py',['diagnose','--output',str(self.output)])
        route=read(self.output/'diagnostic/route.json')['route']
        if route=='warm_start':
            reserve=self.training_reserve(w['elapsed_seconds'],2)+24*180.
            if self.cfg['max_wall_seconds']-self.elapsed()<reserve:raise BudgetStop('insufficient budget for warm start and evaluations')
            self.stage('warm-start','grpo_repair_worker.py',['warm-start','--output',str(self.output)],self.training_reserve(w['elapsed_seconds'],2))
            child=self.child('c');b=self.evaluate(child,'b','weighted_sft',0,initial_policy(read(child/'config.json')))
            gate=b_gate(w,b);atomic_json(self.output/'warm-start/b-gate.json',gate)
            if not gate['passed']:return {'route':route,'stop_reason':'b_gate_rejected','selected_candidate':None,'b_gate':gate}
            results,complete=self.train(['c'],w);selection=select(w,results,b,complete)
        else:
            results,complete=self.train(['plain'] if route=='online_plain' else ['control','curriculum'],w)
            selection=select(w,results,complete=complete)
        return {'route':route,'stop_reason':'completed' if complete else 'completed_budget_stop',**selection}


def resume_spent(output):
    candidates=[read(p).get('elapsed_seconds',0.) for p in (output/'runner_state.json',output/'run_status.json',output/'heartbeat.json') if p.exists()]
    receipt=output/'packaging_receipt.json'
    if receipt.exists():candidates.append(read(receipt)['elapsed_seconds_including_packaging'])
    spent=max(candidates,default=0.)
    if not read(output/'runner_state.json').get('finished'):spent+=5.
    return spent


def package(output,archive,status,study):
    atomic_json(output/'run_status.json',{**status,'status_phase':'before_packaging','elapsed_seconds':study.elapsed()})
    paths=[p for p in output.rglob('*') if p.is_file() and p.suffix not in ('.pt','.pyc')
           and not any(part.startswith('.') for part in p.relative_to(output).parts) and '.writing' not in p.name
           and p.name!='final_status.json']
    start=time.monotonic()
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in tqdm(sorted(paths),desc='Package review',unit='file',file=sys.stdout,disable=False):
            z.write(p,p.relative_to(output.parent).as_posix())
    final={**status,'status_phase':'after_packaging','elapsed_seconds':study.elapsed(),
           'packaging_seconds':time.monotonic()-start,'state_pt_kept_on_cloud':True}
    if final['elapsed_seconds']>study.cfg['max_wall_seconds']:
        final.update(stop_reason='wall_budget_exceeded',selected_candidate=None,error='total wall budget exceeded during packaging')
        if 'selected_warm_start' in final:final['selected_warm_start']=None
    atomic_json(output/'run_status.json',final)
    atomic_json(output/'final_status.json',final)
    atomic_json(output/'attempts'/f'attempt-{study.attempt:03}-final.json',final)
    with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
        z.writestr(output.name+'/final_status.json',json.dumps(final,ensure_ascii=False,indent=2))
        z.writestr(output.name+f'/attempts/attempt-{study.attempt:03}-final.json',json.dumps(final,ensure_ascii=False,indent=2))
    receipt={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':file_hash(archive),
             'elapsed_seconds_including_packaging':study.elapsed(),'budget_seconds':study.cfg['max_wall_seconds']}
    receipt['budget_passed']=receipt['elapsed_seconds_including_packaging']<=receipt['budget_seconds']
    if not receipt['budget_passed']:
        final.update(selected_candidate=None,stop_reason='wall_budget_exceeded',error='wall budget exceeded including archive finalization')
        if 'selected_warm_start' in final:final['selected_warm_start']=None
        atomic_json(output/'run_status.json',final)
        with zipfile.ZipFile(archive,'a',zipfile.ZIP_DEFLATED) as z:
            z.writestr(output.name+'/budget_rejection.json',json.dumps(final,ensure_ascii=False,indent=2))
        receipt.update(bytes=archive.stat().st_size,sha256=file_hash(archive))
    atomic_json(archive.with_suffix('.receipt.json'),receipt)
    atomic_json(output/'packaging_receipt.json',receipt)
    print('Download: '+str(archive),flush=True);print('Receipt: '+str(archive.with_suffix('.receipt.json')),flush=True)
    print('Run status: '+json.dumps({k:final.get(k) for k in ('error','stop_reason','selected_candidate','actual_updates','committed_groups','sft_updates','elapsed_seconds')}),flush=True)
    return final


def run(config,resume=False):
    attempt_started=time.monotonic()
    cfg=read(config)
    if (cfg['schema']!=SCHEMA or cfg['max_wall_seconds']!=21600 or cfg['packaging_reserve_seconds']!=180
            or cfg['k']!=4 or cfg['diagnostic_groups']!=24 or cfg['online_groups']!=36
            or cfg['sft']!={'steps':24,'batch_size':4,'epochs':4,'learning_rate':1e-5,'seed':2026110953,'target_actions':12,'maintenance_actions':12}):
        raise ValueError('frozen master contract differs')
    release=read(ROOT/MANIFEST);verify_files(ROOT,release['files'])
    if config.resolve().relative_to(ROOT).as_posix() not in release['files']:raise ValueError('config not published')
    output=ROOT/'artifacts'/cfg['run_id'];actual=runtime_identity()
    if actual['sqlite']!='3.54.1':raise ValueError('SQLite 3.54.1 required')
    if resume:
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=actual:raise ValueError('resume config/hardware differs')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        if file_hash(ROOT/MANIFEST)!=read(output/'run_manifest.json')['release_sha256']:raise ValueError('resume publication identity differs')
        prior=read(output/'runner_state.json');spent=resume_spent(output);attempt=prior['attempt']+1
        atomic_json(output/'attempts'/f"attempt-{prior['attempt']:03}-runner-snapshot.json",prior)
        if (output/'selection.json').exists() and prior.get('finished') and read(output/'run_status.json').get('error') is None:
            raise ValueError('completed study must not be restarted')
    else:
        if output.exists():raise FileExistsError('run exists; resume the matching unfinished run')
        output.mkdir(parents=True);atomic_json(output/'config.json',cfg);atomic_json(output/'execution_environment.json',actual)
        shutil.copyfile(ROOT/MANIFEST,output/'release_manifest.json')
        sources=source_files();atomic_json(output/'run_manifest.json',{'source_sha256':sources,'release_sha256':file_hash(ROOT/MANIFEST)})
        for rel in sources:
            dest=output/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
        spent,attempt=0.,1
    study=Study(cfg,output,spent,attempt,attempt_started);study.save();thread=threading.Thread(target=study.heartbeat,daemon=True);thread.start()
    archive=ROOT/(cfg['run_id']+f'-review-attempt-{attempt:03}.zip');error=None;selection={}
    try:
        selection=study.execute();atomic_json(output/'selection_proposal.json',selection);atomic_json(output/'selection.json',selection)
        study.stage('root-audit','audit_v2_grpo_repair.py',['--output',str(output)])
    except BudgetStop as exc:
        selection={'selected_candidate':None,'stop_reason':'budget_stop','reason':str(exc)}
        atomic_json(output/'selection.json',selection)
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}';selection={'selected_candidate':None,'stop_reason':'failed'}
        atomic_json(output/'selection.json',selection)
    finally:
        study.stop_event.set();thread.join(timeout=10);study.save(finished=True)
        counts={};counter_errors={}
        for child in (output/'branches').glob('*'):
            if child.is_dir() and not child.name.startswith('.') and child.name!='w':
                try:
                    cursor,state=states(child);counts[child.name]={'groups':cursor,'updates':state['updates']}
                except Exception as exc:counter_errors[child.name]=str(exc)
        sft_steps=len(list((output/'warm-start/checkpoints').glob('step-*')))
        if counter_errors:
            error=error or 'committed counter verification failed'
            selection['selected_candidate']=None
        status={'actual_updates':{n:v['updates'] for n,v in counts.items()},
                'committed_groups':{n:v['groups'] for n,v in counts.items()},'sft_updates':sft_steps,
                'counter_verification_errors':counter_errors,'run_id':cfg['run_id'],'attempt':attempt,'error':error,'max_wall_seconds':21600,'stages':study.stages,**selection}
        final=package(output,archive,status,study)
    if error:raise RuntimeError(error)
    return archive


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/grpo-sql-repair-study-v1.json');p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.config,a.resume)
