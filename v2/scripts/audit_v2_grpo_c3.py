"""Strict matched C3 audit and same-instance paired comparisons."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,verify_files,initial_policy
from grpo_c3_common import matched_configs,states_at,verify_objective_metrics
from audit_v2_grpo import audit as audit_training
from evaluate_v2_grpo import verify_evaluation,verify_closing_evaluation
from apt_v2.grpo_evaluation import paired_evaluation
from apt_v2.grpo_credit import action_advantage
from apt_v2.period_closing import closing_tasks,expert_dependency_proof


def audit(output,local=False):
    from transformers import AutoTokenizer
    cfg,branch_cfg=matched_configs(ROOT/'configs/grpo-c3-matched-v1.json')
    if read(output/'config.json')!=cfg:raise ValueError('matched root config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    children={b:output/'branches'/b for b in ('outcome','fine')};training={};objective={};states={}
    identities=[read(c/'execution_environment.json') for c in children.values()]
    if identities[0]!=identities[1] or read(output/'execution_environment.json')!=identities[0]:raise ValueError('same-instance matched identity differs')
    for b,child in children.items():
        if read(child/'config.json')!=branch_cfg[b]:raise ValueError('frozen matched branch config differs')
        training[b]=audit_training(child,local);objective[b]=verify_objective_metrics(child);states[b]=states_at(child)
        if len(states[b])!=25 or states[b][0]!=initial_policy(branch_cfg[b]):raise ValueError('independent W start and matched group count required')
        if read(child/'closing_dependency_proof.json')!=[expert_dependency_proof(t) for t in closing_tasks()]:raise ValueError('SQLite dependency proof differs')
    evals={};summaries={};checks={};extensions={}
    for branch,label,key,cursor in [('outcome','weighted_sft','weighted_sft',0),('outcome','grpo_g0024','outcome',24),('fine','grpo_g0024','fine',24)]:
        child=children[branch];tok=AutoTokenizer.from_pretrained(child/'tokenizer',local_files_only=True,trust_remote_code=False)
        spec=read(child/'evaluations'/(label+'-policy.json'))
        if spec!={'label':label,'group_cursor':cursor,'policy':states[branch][cursor]}:raise ValueError('matched evaluation cursor differs')
        check=verify_evaluation(child,label,spec,tok,local=local)
        if not check['passed']:raise ValueError('matched evaluation quality gate failed')
        extensions[key]=verify_closing_evaluation(child,label,spec,tok);checks[key]={k:v for k,v in check.items() if k!='calendar'}
        data=read(child/'evaluations'/(label+'.json'));evals[key]=data['episodes'];summaries[key]=check['summary']
    comparisons={'outcome_vs_W':paired_evaluation(evals['weighted_sft'],evals['outcome']),'fine_vs_W':paired_evaluation(evals['weighted_sft'],evals['fine']),'fine_vs_outcome':paired_evaluation(evals['outcome'],evals['fine'])}
    baseline=summaries['weighted_sft']['sample']['complete'];primary='recovery' if baseline['successes']==baseline['episodes'] else 'complete'
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_files_and_state','training':training,'independent_objective_recomputation':objective,'evaluation':checks,'paired':comparisons,'summaries':summaries,'closing_development':extensions,'primary_descriptive_metric':primary,'acceptance_thresholds':None,'C4_new_task_confirmation':'pending','hardware':identities[0]}
    curve=[];credits=[];evaluation_curve=[]
    for branch,child in children.items():
        outcome_zero=0
        for i in range(24):
            g=read(child/'groups'/f'group-{i:04}.json');tr=read(child/'transitions'/f'group-{i:04}.json');c=g['credit']
            outcome_zero+=bool(c.get('outcome_zero_variance',c['zero_variance']))
            curve.append({'branch':branch,'group':i,'kind':g['spec']['kind'],'reward_mean':c['mean'],'reward_std':c['population_std'],'eligible':c['eligible'],'updated':tr['updated'],'actual_updates':tr['after']['updates'],'actions':g['actions'],'sampled_k3':tr['metrics'].get('sampled_k3',0),'grad_norm':tr['metrics'].get('grad_norm',0)})
            for j,e in enumerate(g['episodes']):
                for relative,row in enumerate(e['turns']):credits.append({'branch':branch,'group':i,'episode':j,'turn':row['turn'],'advantage':action_advantage(c,j,relative),'return_to_go':c['turn_returns'][j][relative] if 'turn_returns' in c else e['return'],'process_reward':row['process_reward']})
        training[branch]['outcome_zero_variance_groups']=outcome_zero
    for branch,modes in summaries.items():
        for mode,kinds in modes.items():
            for kind,v in kinds.items():evaluation_curve.append({'branch':branch,'mode':mode,'kind':kind,'successes':v['successes'],'episodes':v['episodes'],'actions':v['actions'],'response_tokens':v['response_tokens']})
    for name,rows in [('training_curve.csv',curve),('action_credit.csv',credits),('evaluation_curve.csv',evaluation_curve)]:
        with (output/(('local_' if local else '')+name)).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(output/('local_c3_audit.json' if local else 'c3_audit.json'),result)
    atomic_json(output/('local_comparison_summary.json' if local else 'comparison_summary.json'),{'summaries':summaries,'paired':comparisons,'closing_development':extensions,'primary_descriptive_metric':primary})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local',action='store_true');a=p.parse_args();audit(a.output,a.local)
