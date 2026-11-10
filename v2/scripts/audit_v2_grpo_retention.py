"""Replay retention recipe, fresh W/B evaluations and optional GRPO selection."""
import argparse,math,sys
from pathlib import Path
from tqdm import tqdm
from grpo_common import ROOT,read,atomic_json,file_hash,verify_files,verify_inputs
from grpo_repair_worker import audit_sft
from run_v2_grpo_repair import evaluation,states,select
from apt_v2.grpo_repair_study import b_gate
from apt_v2.grpo_retention_study import validate_master,validate_prior


def audit(output,local=False):
    cfg=read(output/'config.json');validate_master(cfg);prior=read(output/'prior-evidence.json');validate_prior(prior,cfg)
    if prior!=read(ROOT/cfg['prior_evidence']):raise ValueError('prior evidence differs')
    manifest=read(output/'run_manifest.json');verify_files(ROOT,manifest['source_sha256'])
    if file_hash(output/'release_manifest.json')!=manifest['release_sha256']:raise ValueError('publication identity differs')
    verify_files(ROOT,read(output/'release_manifest.json')['files']);selection=read(output/'selection.json')
    prepared=[p for p in (output/'branches').iterdir() if p.is_dir() and not p.name.startswith('.')]
    with tqdm(total=len(prepared)+2,desc='Retention study replay',unit='stage',file=sys.stdout,disable=False) as bar:
        for child in prepared:
            verify_inputs(child,runtime=not local,local=local)
            if read(child/'execution_environment.json')!=read(output/'execution_environment.json'):
                raise ValueError('within-study runtime identity differs')
            bar.update(1)
        w=evaluation(output/'branches/w','weighted_sft',local);bar.update(1)
        if not w['evaluation_quality_passed']:
            expected={'route':'warm_start','stop_reason':'baseline_quality_rejected','selected_candidate':None}
            result={'passed':True,'selected_candidate':None,'baseline_quality_rejected':True}
            bar.update(1)
        else:
            sft=audit_sft(output,local);b=evaluation(output/'branches/c','weighted_sft',local)
            gate=b_gate(w,b)
            if gate!=read(output/'warm-start/b-gate.json'):raise ValueError('B gate differs')
            bar.update(1)
            if not gate['passed']:
                expected={'route':'warm_start','stop_reason':'b_gate_rejected','selected_candidate':None,'b_gate':gate}
                result={'passed':True,'selected_candidate':None,'b_gate':gate,'sft':sft}
            else:
                bar.total+=1;bar.refresh()
                from audit_v2_grpo import audit as audit_branch
                from grpo_c3_common import verify_objective_metrics
                from run_v2_grpo_retention import Study
                child=output/'branches/c';cursor,state=states(child);branch=audit_branch(child,local)
                if not branch['passed']:raise ValueError('C evidence audit failed')
                objective=verify_objective_metrics(child)
                if cursor%2:raise ValueError('unpaired training boundary')
                if cursor<36:
                    reserve=Study(cfg,output).training_reserve(w['elapsed_seconds'],1)
                    decisions=[read(p) for p in (output/'budget-decisions').glob('*.json')]
                    if not any(d['cycle']==cursor and d['continue'] is False and math.isfinite(d['elapsed_seconds'])
                               and 0<=d['elapsed_seconds']<=16000 and d['reserved_final_seconds']==reserve
                               and d['next_cycle_seconds']>=300 and 16000-d['elapsed_seconds']<reserve+d['next_cycle_seconds']
                               for d in decisions):raise ValueError('paired resource stop evidence missing')
                c={**evaluation(child,f'grpo_g{cursor:04}',local),
                   'effective_bf16_change':bool(branch.get('effective_bf16_tensor_hash_changed',False))}
                expected={'route':'warm_start','stop_reason':'completed' if cursor==36 else 'completed_budget_stop',
                          **select(w,{'c':c},b,cursor==36)}
                result={'passed':True,'selected_candidate':expected['selected_candidate'],'b_gate':gate,'sft':sft,
                        'training_groups':cursor,'actual_updates':state['updates'],'objective_recomputation':objective}
                bar.update(1)
        if selection!=expected:raise ValueError('selection recomputation differs')
    result.update(scope='local_returned_files' if local else 'cloud_with_state',new_verifier='sql_semantics_v2',
                  prior_elapsed_seconds=cfg['prior_elapsed_seconds'],current_budget_seconds=16000,total_authorized_seconds=21600)
    atomic_json(output/('local_audit.json' if local else 'audit.json'),result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local',action='store_true')
    a=p.parse_args();audit(a.output,a.local)
