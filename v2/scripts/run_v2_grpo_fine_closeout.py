"""Run and audit the matched fine GRPO closeout-credit experiment."""
import argparse
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from tqdm import tqdm

from grpo_common import ROOT, atomic_json, file_hash, initial_policy, load_transition, read, runtime_identity, verify_files
from apt_v2 import grpo_fine_closeout as study
from apt_v2.grpo_lr_midpoint import completion_diagnostics
from run_v2_grpo import source_files
from run_v2_grpo_repair import BudgetStop, Study, evaluation, package, resume_spent, states


def branch_config(name, retained):
    template = read(ROOT/study.TEMPLATE)
    method_name = name.removeprefix('confirm-')
    cfg = {**template, 'run_id': f'{study.RUN_ID}-{name}',
           'credit_method': study.METHODS.get(method_name, study.LOCAL),
           'closeout_states': list(retained) if name in study.BRANCHES else [],
           'evaluation_panel': 'confirmation' if name.startswith('confirm-') else 'development'}
    study.validate_branch(cfg)
    return cfg


def persist(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'committed {path.name} differs')
    if not path.exists():
        atomic_json(path, value)
    return value


def recovery_pair(local, closeout):
    def key(e):
        return tuple(e[k] for k in ('task_id', 'scenario', 'rollout_index', 'seed'))
    def rows(result):
        episodes = [e for e in result['episodes'] if e['evaluation_mode']=='sample' and e['scenario'] is not None]
        indexed = {key(e): e for e in episodes}
        if len(episodes)!=84 or len(indexed)!=84:
            raise ValueError('84 distinct sampled recovery episodes required')
        return indexed
    a,b = rows(local),rows(closeout)
    if a.keys()!=b.keys() or any((e['start_id'], e['sampling']) != (b[k]['start_id'], b[k]['sampling'])
                                  for k,e in a.items()):
        raise ValueError('credit-arm recovery pair differs')
    details = {}
    for k,e in a.items():
        task = details.setdefault(e['task_id'], {'local':0,'closeout':0,'gained':0,'lost':0})
        new = b[k]['success']
        task['local'] += e['success']
        task['closeout'] += new
        task['gained'] += new and not e['success']
        task['lost'] += e['success'] and not new
    return {'by_task': details, **{name: sum(t[name] for t in details.values())
                                   for name in ('local','closeout','gained','lost')}}


def result_record(w, results):
    decision = study.selection(w, results)
    return {**decision, 'paired_recovery': recovery_pair(results['local'], results['closeout']) if results else None,
            'completion': {name: completion_diagnostics(w, result, name) for name,result in results.items()}}


class FineStudy(Study):
    def stage(self, name, script, args, tail_reserve=180., stage_timeout=None):
        if 'confirm-' in name:
            tail_reserve = max(tail_reserve, 600.)
        return super().stage(name, script, args, tail_reserve, stage_timeout)

    def child(self, name, retained=()):
        cfg = branch_config(name, list(retained))
        persist(self.output/'branch-configs'/(name+'.json'), cfg)
        self.stage('prepare-'+name, 'grpo_repair_worker.py', ['prepare','--output',str(self.output),'--name',name])
        child = self.output/'branches'/name
        if read(child/'execution_environment.json') != read(self.output/'execution_environment.json'):
            raise ValueError('branch runtime differs from frozen job runtime')
        return child

    def preflight(self, wchild):
        from transformers import AutoTokenizer
        from grpo_repair_worker import finalize_diagnostic
        route = self.output/'diagnostic/route.json'
        if route.exists():
            return read(route)
        timer = self.output/'diagnostic/deadline.json'
        deadline = read(timer)['elapsed_deadline'] if timer.exists() else self.elapsed()+study.PREFLIGHT_CAP
        persist(timer, {'elapsed_deadline':deadline})
        remaining = deadline-self.elapsed()-15
        if remaining>0:
            try:
                self.stage('w-preflight','grpo_repair_worker.py',['diagnose','--output',str(self.output)],
                           study.FINAL_RESERVE, stage_timeout=remaining)
            except subprocess.TimeoutExpired:
                pass
            if route.exists():
                return read(route)
        tok = AutoTokenizer.from_pretrained(wchild/'tokenizer',local_files_only=True,trust_remote_code=False)
        return finalize_diagnostic(self.output, tok, allow_partial=True)

    def train_pair(self, children, w):
        reserve = max(study.FINAL_RESERVE-w['elapsed_seconds'], 2*max(2400.,1.25*w['elapsed_seconds'])+900.)
        cursors = {name:states(child) for name,child in children.items()}
        stop = self.output/'training_stop.json'
        if stop.exists():
            done = read(stop)['groups_per_arm']
            if any(cursor!=done for cursor,_ in cursors.values()):
                raise ValueError('saved paired stop cursor differs')
            return cursors
        with tqdm(total=2*study.GROUPS, initial=sum(c for c,_ in cursors.values()),
                  desc='Matched fine GRPO',unit='group',file=sys.stdout,disable=False) as bar:
            for block in range(0,study.GROUPS,study.BLOCK):
                if all(c>=block+study.BLOCK for c,_ in cursors.values()):
                    continue
                if any(not block<=c<=block+study.BLOCK for c,_ in cursors.values()):
                    raise ValueError('resumed training outside one paired block')
                timing = [s['elapsed_seconds'] for s in self.stages if s['stage'].startswith(('collect-','update-'))]
                seconds_per_group = max(150.,2.5*sum(timing)/len(timing)) if timing else 180.
                estimate = sum(block+study.BLOCK-c for c,_ in cursors.values())*seconds_per_group
                remaining = self.cfg['max_wall_seconds']-self.elapsed()
                enough = remaining >= reserve+estimate
                atomic_json(self.output/'budget-decisions'/f'{self.attempt:03}-{block:04}.json',
                            {'block':block,'remaining_seconds':remaining,
                             'reserved_seconds':reserve,'estimate_seconds':estimate,'continue':enough})
                if not enough:
                    break
                for index in range(block,block+study.BLOCK):
                    for name,child in children.items():
                        cursor,state = cursors[name]
                        if cursor>index:
                            continue
                        if cursor!=index:
                            raise ValueError('paired training cursor differs')
                        policy_file = child/'policies'/f'before-group-{index:04}.json'
                        persist(policy_file,state)
                        args=['--output',str(child),'--group-index',str(index),'--policy-file',str(policy_file)]
                        self.stage(f'collect-{name}-{index:04}','grpo_worker.py',['collect',*args],reserve)
                        self.stage(f'update-{name}-{index:04}','grpo_worker.py',['update',*args],reserve)
                        state = load_transition(child,index,state)['after']
                        cursors[name]=(index+1,state)
                        atomic_json(child/'runner_state.json',{'policy':state,'next_group':index+1,
                                                               'refresh_only_last_group':False})
                        bar.update(1)
        counts = {c for c,_ in cursors.values()}
        done = min(counts)
        if len(counts)!=1 or done%study.BLOCK or done==0:
            raise BudgetStop('no completed paired training schedule available for evaluation')
        persist(stop, {'groups_per_arm':done,'reason':'group_cap' if done==study.GROUPS else 'budget_reserve'})
        return cursors

    def confirm(self, name, w, winner, trained_child, cursor, state):
        if name is None:
            return {'status':'no_candidate'}
        plan = self.output/'confirmation-plan.json'
        if plan.exists():
            if read(plan)['candidate']!=name:
                raise ValueError('confirmation candidate changed')
            if not read(plan)['execute']:
                return {'status':'budget_reserve','candidate':name}
        else:
            need = max(4800.,1.20*(w['elapsed_seconds']+winner['elapsed_seconds'])+900.)
            remaining = self.cfg['max_wall_seconds']-self.elapsed()
            execute = remaining>=need
            persist(plan,{'candidate':name,'execute':execute,'estimated_seconds':need,'remaining_seconds':remaining})
            if not execute:
                return {'status':'budget_reserve','candidate':name}
        try:
            left = self.child('confirm-w')
            right = self.child('confirm-'+name)
            destination = right/state['checkpoint']/'adapter'
            if not destination.exists():
                temporary = destination.with_name('.adapter-copying')
                temporary = temporary.with_name(temporary.name+'-'+str(time.time_ns()))
                shutil.copytree(trained_child/state['checkpoint']/'adapter',temporary)
                temporary.rename(destination)
            if file_hash(destination/'adapter_model.safetensors')!=state['adapter_sha256']:
                raise ValueError('confirmation adapter differs from selected final checkpoint')
            ev_w=self.evaluate(left,'confirm-w','weighted_sft',0,initial_policy(read(left/'config.json')))
            ev_b=self.evaluate(right,'confirm-'+name,f'grpo_g{cursor:04}',cursor,state)
            ev_b['effective_bf16_change'] = winner['effective_bf16_change']
            return {'status':'complete','candidate':name,'decision':study.selection(ev_w,{name:ev_b})}
        except (BudgetStop, subprocess.TimeoutExpired) as exc:
            return {'status':'budget_stop','candidate':name,'reason':str(exc)}

    def execute(self):
        wchild = self.child('w')
        w = self.evaluate(wchild,'w','weighted_sft',0,initial_policy(read(wchild/'config.json')))
        if not w['evaluation_quality_passed']:
            return {**result_record(w,{}),'confirmation':{'status':'no_candidate'},'stop_reason':'baseline_quality_rejected'}
        preflight = self.preflight(wchild)
        children = {name:self.child(name,preflight['retained']) for name in study.BRANCHES}
        cursors = self.train_pair(children,w)
        results = {}
        for name,child in children.items():
            cursor,state = cursors[name]
            policy_file = child/'policies'/f'before-group-{cursor:04}.json'
            persist(policy_file,state)
            atomic_json(child/'runner_state.json',{'policy':state,'next_group':cursor,'refresh_only_last_group':True})
            self.stage('refresh-'+name,'grpo_worker.py',['collect','--output',str(child),'--group-index',str(cursor),
                                                        '--policy-file',str(policy_file)])
            self.stage('audit-'+name,'audit_v2_grpo.py',['--output',str(child)])
            results[name] = self.evaluate(child,name,f'grpo_g{cursor:04}',cursor,state)
            results[name]['effective_bf16_change'] = bool(read(child/'audit.json').get('effective_bf16_tensor_hash_changed'))
        decision = result_record(w,results)
        chosen = decision['selected_candidate']
        confirm = self.confirm(chosen,w,results[chosen],children[chosen],*cursors[chosen]) if chosen else {'status':'no_candidate'}
        return {**decision,'preflight':preflight,'confirmation':confirm,'stop_reason':read(self.output/'training_stop.json')['reason']}


def audit(output, local=False):
    from transformers import AutoTokenizer
    from audit_v2_grpo import audit as audit_branch
    from grpo_repair_worker import verify_diagnostic
    from grpo_c3_common import verify_objective_metrics
    manifest = read(output/'run_manifest.json')
    verify_files(ROOT,manifest['source_sha256'])
    verify_files(output/'source',manifest['source_sha256'])
    if file_hash(output/'release_manifest.json') != manifest['release_sha256']:
        raise ValueError('returned release differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files'])
    study.validate_master(read(output/'config.json'))
    recorded = read(output/'selection.json')
    wchild = output/'branches/w'
    if read(wchild/'config.json') != branch_config('w',[]):
        raise ValueError('W branch differs')
    w = evaluation(wchild,'weighted_sft',local)
    if w['spec']['policy'] != initial_policy(read(wchild/'config.json')):
        raise ValueError('baseline evaluated with unexpected policy')
    results = {}
    cursors = {}
    objectives = {}
    extras = {'confirmation':{'status':'no_candidate'},'stop_reason':'baseline_quality_rejected'}
    if w['evaluation_quality_passed']:
        tok = AutoTokenizer.from_pretrained(wchild/'tokenizer',local_files_only=True,trust_remote_code=False)
        groups,preflight = verify_diagnostic(output,tok,allow_partial=True)
        if preflight != read(output/'diagnostic/route.json'):
            raise ValueError('committed preflight result differs')
        for g in groups:
            if read(output/'diagnostic/groups'/f"{g['spec']['index']:04}.json") != g:
                raise ValueError('committed preflight group differs')
        stopped = read(output/'training_stop.json')
        done = stopped['groups_per_arm']
        if stopped['reason'] != ('group_cap' if done == study.GROUPS else 'budget_reserve'):
            raise ValueError('paired stop reason differs')
        if done < study.GROUPS:
            decisions = sorted((output/'budget-decisions').glob(f'*-{done:04}.json'))
            last = read(decisions[-1]) if decisions else {}
            if (last.get('continue') is not False or last.get('block') != done
                    or last['remaining_seconds'] >= last['reserved_seconds']+last['estimate_seconds']):
                raise ValueError('paired budget stop lacks a matching decision')
        for name in study.BRANCHES:
            child = output/'branches'/name
            if read(child/'config.json') != branch_config(name,preflight['retained']):
                raise ValueError('credit branch differs from audited preflight')
            cursor,state = states(child)
            if cursor!=stopped['groups_per_arm'] or cursor%study.BLOCK or not 0<cursor<=study.GROUPS:
                raise ValueError('paired audit stop differs')
            evidence = audit_branch(child,local=True) if local else read(child/'audit.json')
            if not evidence['passed'] or evidence['final_policy']!=state or evidence['attempted_groups']!=cursor:
                raise ValueError('branch replay or state differs')
            cursors[name]=(cursor,state)
            objectives[name] = verify_objective_metrics(child)
            if objectives[name]['actual_updates_recomputed'] != state['updates']:
                raise ValueError('objective replay update count differs')
            results[name]=evaluation(child,f'grpo_g{cursor:04}',local)
            results[name]['effective_bf16_change'] = bool(evidence.get('effective_bf16_tensor_hash_changed'))
            if results[name]['spec']['policy']!=state:
                raise ValueError('candidate evaluation did not use final policy')
        extras = {'preflight':preflight,'confirmation':recorded['confirmation'],'stop_reason':stopped['reason']}
    expected = {**result_record(w,results),**extras}
    confirmation = expected['confirmation']
    name = expected['selected_candidate']
    if name is None:
        if confirmation != {'status':'no_candidate'} or (output/'confirmation-plan.json').exists():
            raise ValueError('confirmation requires a qualified candidate')
    else:
        plan = read(output/'confirmation-plan.json')
        required = max(4800.,1.20*(w['elapsed_seconds']+results[name]['elapsed_seconds'])+900.)
        if (confirmation.get('candidate') != name or plan['candidate'] != name
                or plan['estimated_seconds'] != required or plan['execute'] != (plan['remaining_seconds'] >= required)
                or confirmation['status'] not in (('complete','budget_stop') if plan['execute'] else ('budget_reserve',))):
            raise ValueError('confirmation budget or selection differs')
    if confirmation['status']=='complete':
        cursor,state=cursors[name]
        a,b=output/'branches/confirm-w',output/'branches'/('confirm-'+name)
        if read(a/'config.json')!=branch_config('confirm-w',[]) or read(b/'config.json')!=branch_config('confirm-'+name,[]):
            raise ValueError('independent panel configuration differs')
        ew,eb=evaluation(a,'weighted_sft',local),evaluation(b,f'grpo_g{cursor:04}',local)
        if ew['spec']['policy']!=initial_policy(read(a/'config.json')) or eb['spec']['policy']!=state:
            raise ValueError('confirmation checkpoint binding differs')
        eb['effective_bf16_change'] = results[name]['effective_bf16_change']
        if confirmation['decision']!=study.selection(ew,{name:eb}):
            raise ValueError('confirmation acceptance differs')
    if recorded!=expected:
        raise ValueError('root fine closeout result differs')
    summary={'passed':True,'scope':'local_returned_files' if local else 'cloud_with_state',
             'actual_updates':{n:s['updates'] for n,(_,s) in cursors.items()},'objective_replay':objectives,'selection':expected}
    atomic_json(output/('local_audit.json' if local else 'audit.json'),summary)
    return summary


def run(config, resume=False):
    started = time.monotonic()
    cfg=read(config)
    study.validate_master(cfg)
    release=ROOT/study.MANIFEST
    verify_files(ROOT,read(release)['files'])
    if config.resolve().relative_to(ROOT).as_posix()!=study.CONFIG:
        raise ValueError('run the frozen master config')
    output=ROOT/'artifacts'/study.RUN_ID
    runtime=runtime_identity()
    if runtime['sqlite']!='3.54.1':
        raise ValueError('SQLite 3.54.1 required')
    if resume:
        previous=read(output/'runner_state.json')
        if read(output/'config.json')!=cfg or read(output/'execution_environment.json')!=runtime:
            raise ValueError('resume config/hardware differs')
        verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
        if read(output/'run_manifest.json')['release_sha256']!=file_hash(release):
            raise ValueError('resume release differs')
        if previous.get('finished') and read(output/'run_status.json').get('error') is None:
            raise ValueError('completed study must not restart')
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
            dest=output/'source'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/rel,dest)
        spent,attempt=0.,1
    runner=FineStudy(cfg,output,spent,attempt,started)
    runner.save()
    heartbeat=threading.Thread(target=runner.heartbeat,daemon=True)
    heartbeat.start()
    error=None
    result={'selected_candidate':None,'formal_baseline':'W'}
    archive=ROOT/(study.RUN_ID+f'-review-attempt-{attempt:03}.zip')
    try:
        result=runner.execute()
        atomic_json(output/'selection.json',result)
        runner.stage('root-audit','run_v2_grpo_fine_closeout.py',['--audit',str(output)])
    except BudgetStop as exc:
        result={'selected_candidate':None,'formal_baseline':'W','stop_reason':'budget_stop','reason':str(exc)}
    except (Exception,KeyboardInterrupt) as exc:
        error=f'{type(exc).__name__}: {exc}'
        result={'selected_candidate':None,'formal_baseline':'W','stop_reason':'failed'}
    finally:
        runner.stop_event.set()
        heartbeat.join(timeout=10)
        runner.save(finished=True)
        counts, groups = {}, {}
        for name in study.BRANCHES:
            child=output/'branches'/name
            if (child/'config.json').exists():
                try:
                    groups[name],state=states(child)
                    counts[name]=state['updates']
                except Exception as exc:
                    error=error or f'counter audit failed: {exc}'
        if error:
            result={'selected_candidate':None,'formal_baseline':'W','stop_reason':'failed'}
        status={'run_id':study.RUN_ID,'attempt':attempt,'error':error,'actual_updates':counts,
                'committed_groups':groups,'sft_updates':0,'max_wall_seconds':study.CAP,'stages':runner.stages,**result}
        package(output,archive,status,runner)
    if error:
        raise RuntimeError(error)
    return archive


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/study.CONFIG)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--local',action='store_true')
    args=parser.parse_args()
    if args.audit:
        audit(args.audit,args.local)
    else:
        run(args.config,args.resume)
