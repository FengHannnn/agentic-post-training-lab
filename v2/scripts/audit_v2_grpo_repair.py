"""Independent replay and candidate selection audit for the gated SQL-repair study."""
import argparse,sys
from pathlib import Path
from tqdm import tqdm
from grpo_common import read,atomic_json,verify_files,verify_inputs,file_hash
from grpo_repair_worker import verify_diagnostic,audit_sft
from run_v2_grpo_repair import states,evaluation,select
from apt_v2.grpo_repair_study import b_gate


def audit(output,local=False):
    from transformers import AutoTokenizer
    from audit_v2_grpo import audit as audit_branch
    cfg=read(output/'config.json');selection=read(output/'selection.json')
    from grpo_common import ROOT
    manifest=read(output/'run_manifest.json')
    verify_files(ROOT,manifest['source_sha256'])
    if file_hash(output/'release_manifest.json')!=manifest['release_sha256']:raise ValueError('frozen publication manifest differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files'])
    prepared=[p for p in (output/'branches').iterdir() if p.is_dir() and not p.name.startswith('.')]
    with tqdm(total=len(prepared)+2,desc='Repair study replay',unit='stage',file=sys.stdout,disable=False) as bar:
        for child in prepared:
            verify_inputs(child,runtime=not local,local=local)
            if read(child/'execution_environment.json')!=read(output/'execution_environment.json'):
                raise ValueError('within-study hardware identity differs')
            bar.update(1)
        wchild=output/'branches/w';w=evaluation(wchild,'weighted_sft',local)
        if selection['stop_reason']=='baseline_quality_rejected':
            if w['evaluation_quality_passed']:raise ValueError('baseline quality rejection differs')
            result={'passed':True,'selected_candidate':None,'baseline_quality_rejected':True}
        else:
            tok=AutoTokenizer.from_pretrained(wchild/'tokenizer',local_files_only=True,trust_remote_code=False)
            groups,route=verify_diagnostic(output,tok);bar.update(1)
            if route['route']!=selection['route']:raise ValueError('route recomputation differs')
            b=None
            if route['route']=='warm_start':
                audit_sft(output,local);b=evaluation(output/'branches/c','weighted_sft',local)
                gate=b_gate(w,b)
                if gate!=read(output/'warm-start/b-gate.json'):raise ValueError('B gate differs')
                if not gate['passed']:
                    if selection!={'route':'warm_start','stop_reason':'b_gate_rejected','selected_candidate':None,'b_gate':gate}:
                        raise ValueError('B rejection selection differs')
                    result={'passed':True,'selected_candidate':None,'b_gate':gate,'diagnostic_groups':24,
                            'scope':'local_returned_files' if local else 'cloud_with_state','new_verifier':'sql_semantics_v2'}
                    atomic_json(output/('local_audit.json' if local else 'audit.json'),result);return result
                names=['c']
            else:names=['plain'] if route['route']=='online_plain' else ['control','curriculum']
            results={};counts=[];branch_audits={}
            for name in names:
                child=output/'branches'/name;cursor,state=states(child);counts.append(cursor)
                branch_audits[name]=audit_branch(child,local)
                from grpo_c3_common import verify_objective_metrics
                branch_audits[name]['objective_recomputation']=verify_objective_metrics(child)
                if not branch_audits[name]['passed']:raise ValueError('branch evidence audit failed')
                results[name]={**evaluation(child,f'grpo_g{cursor:04}',local),
                               'effective_bf16_change':bool(branch_audits[name].get('effective_bf16_tensor_hash_changed',False))}
            if len(set(counts))!=1 or counts[0]%2:raise ValueError('paired training exposure differs')
            if counts[0]<36:
                import math
                from run_v2_grpo_repair import Study
                decisions=[read(p) for p in (output/'budget-decisions').glob('*.json')]
                stopped=[d for d in decisions if d['cycle']==counts[0] and d['continue'] is False]
                reserve=Study(cfg,output).training_reserve(w['elapsed_seconds'],len(names))
                if not any(math.isfinite(d['elapsed_seconds']) and 0<=d['elapsed_seconds']<=21600
                           and d['reserved_final_seconds']==reserve and d['next_cycle_seconds']>=300*len(names)
                           and 21600-d['elapsed_seconds']<reserve+d['next_cycle_seconds'] for d in stopped):
                    raise ValueError('paired budget stop lacks reserved-work evidence')
            expected={'route':route['route'],'stop_reason':'completed' if counts[0]==36 else 'completed_budget_stop',
                      **select(w,results,b,counts[0]==36)}
            if selection!=expected:raise ValueError('candidate selection recomputation differs')
            result={'passed':True,'selected_candidate':expected['selected_candidate'],'diagnostic_groups':len(groups),
                    'diagnostic_episodes':sum(len(g['episodes']) for g in groups),'training_groups':dict(zip(names,counts)),
                    'actual_updates':{n:branch_audits[n]['actual_updates'] for n in names},'selection':expected}
        bar.update(1)
    result.update(scope='local_returned_files' if local else 'cloud_with_state',new_verifier='sql_semantics_v2')
    atomic_json(output/('local_audit.json' if local else 'audit.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local',action='store_true');a=p.parse_args();audit(a.output,a.local)
