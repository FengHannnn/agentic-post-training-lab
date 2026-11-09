"""Replay both outcome branches and separate standard selection from targeted diagnostics."""
import argparse
import csv
from pathlib import Path
from grpo_common import ROOT,read,atomic_json,verify_files,initial_policy,file_hash
from grpo_curriculum_common import matched_configs,states_at,verify_objective_metrics,select_candidate,BRANCHES
from audit_v2_grpo import audit as audit_training
from evaluate_v2_grpo import verify_evaluation,verify_targeted_evaluation
from apt_v2.grpo_evaluation import paired_evaluation,evaluation_key
from apt_v2.grpo_curriculum import frozen_inputs,TARGET_SCENARIOS


def targeted_pairs(before,after):
    a={evaluation_key(e):e['success'] for e in before};b={evaluation_key(e):e['success'] for e in after}
    if len(a)!=len(before) or len(b)!=len(after) or a.keys()!=b.keys():raise ValueError('targeted paired keys differ')
    def stats(keys):return {'episodes':len(keys),'improved':sum(not a[k] and b[k] for k in keys),
        'regressed':sum(a[k] and not b[k] for k in keys),'delta_successes':sum(int(b[k])-int(a[k]) for k in keys)}
    return {mode:{**stats([k for k in a if k[2]==mode]),'by_scenario':{
        s:stats([k for k in a if k[2]==mode and k[1]==s]) for s in TARGET_SCENARIOS}} for mode in ('greedy','sample')}


def audit(output,local=False):
    from transformers import AutoTokenizer
    cfg,branch_cfg=matched_configs(ROOT/'configs/grpo-outcome-curriculum-v1.json')
    if read(output/'config.json')!=cfg:raise ValueError('curriculum root config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    children={b:output/'branches'/b for b in BRANCHES};training={};objective={};states={}
    identities=[read(c/'execution_environment.json') for c in children.values()]
    if identities[0]!=identities[1] or read(output/'execution_environment.json')!=identities[0]:
        raise ValueError('matched GPU/software differs')
    count=cfg['groups_per_branch']
    for b,child in children.items():
        if read(child/'config.json')!=branch_cfg[b]:raise ValueError('frozen branch config differs')
        for name,value in frozen_inputs(branch_cfg[b]).items():
            if read(child/name)!=value:raise ValueError('frozen curriculum input differs: '+name)
        training[b]=audit_training(child,local);objective[b]=verify_objective_metrics(child);states[b]=states_at(child)
        if len(states[b])!=count+1 or states[b][0]!=initial_policy(branch_cfg[b]):
            raise ValueError('independent W start and fixed matched group count required')
        if objective[b]['actual_updates_recomputed']!=training[b]['actual_updates']:
            raise ValueError('objective audit update coverage differs')
    evals={};targeted={};summaries={};checks={};target_checks={}
    for branch,label,key,cursor in [('control','weighted_sft','weighted_sft',0),
                                   ('control',f'grpo_g{count:04}','control',count),
                                   ('lr10x',f'grpo_g{count:04}','lr10x',count)]:
        child=children[branch]
        tok=AutoTokenizer.from_pretrained(child/'tokenizer',local_files_only=True,trust_remote_code=False)
        spec=read(child/'evaluations'/(label+'-policy.json'))
        if spec!={'label':label,'group_cursor':cursor,'policy':states[branch][cursor]}:raise ValueError('evaluation cursor differs')
        check=verify_evaluation(child,label,spec,tok,local=local)
        if not check['passed']:raise ValueError('evaluation quality gate failed')
        target_checks[key]=verify_targeted_evaluation(child,label,spec,tok)
        checks[key]={k:v for k,v in check.items() if k!='calendar'}
        data=read(child/'evaluations'/(label+'.json'))
        evals[key]=data['episodes'];targeted[key]=data['targeted_episodes'];summaries[key]=check['summary']
    comparisons={b+'_vs_W':paired_evaluation(evals['weighted_sft'],evals[b],branch_cfg[b]) for b in BRANCHES}
    comparisons['lr10x_vs_control']=paired_evaluation(evals['control'],evals['lr10x'],branch_cfg['control'])
    target_pairs={b+'_vs_W':targeted_pairs(targeted['weighted_sft'],targeted[b]) for b in BRANCHES}
    result={'passed':True,'scope':'local_returned_files' if local else 'cloud_files_and_state',
        'training':training,'independent_objective_recomputation':objective,'evaluation':checks,
        'paired':comparisons,'summaries':summaries,'targeted_evaluation':target_checks,
        'targeted_paired':target_pairs,'development_selection':select_candidate(summaries),
        'extra_supervised_updates':0,'extra_dpo_updates':0,'reference':'independent_W_per_branch',
        'hardware':identities[0]}
    curve=[];coverage=[];evaluation=[]
    for branch,child in children.items():
        for i in range(count):
            g=read(child/'groups'/f'group-{i:04}.json');tr=read(child/'transitions'/f'group-{i:04}.json');c=g['credit']
            curve.append({'branch':branch,'learning_rate':branch_cfg[branch]['learning_rate'],'group':i,
                'task_id':g['spec']['task_id'],'scenario':g['spec']['scenario'],'kind':g['spec']['kind'],
                'reward_mean':c['mean'],'reward_std':c['population_std'],'eligible':c['eligible'],
                'updated':tr['updated'],'skip_reason':tr['metrics']['skip_reason'],'actual_updates':tr['after']['updates'],
                'actions':g['actions'],'response_tokens':g['response_tokens'],'sampled_k3':tr['metrics'].get('sampled_k3',0),
                'clip_fraction':tr['metrics'].get('clip_fraction',0),'grad_norm':tr['metrics'].get('grad_norm',0),
                'adapter_delta_l2':tr['metrics'].get('adapter_delta_l2',0)})
        for scenario in [None,*branch_cfg[branch]['recovery_scenarios']]:
            rows=[r for r in curve if r['branch']==branch and r['scenario']==scenario]
            coverage.append({'branch':branch,'scenario':scenario,'groups':len(rows),
                'eligible_groups':sum(r['eligible'] for r in rows),'actual_updates':sum(r['updated'] for r in rows)})
    for branch,modes in summaries.items():
        for mode,kinds in modes.items():
            for kind,v in kinds.items():evaluation.append({'branch':branch,'mode':mode,'kind':kind,
                'successes':v['successes'],'episodes':v['episodes'],'actions':v['actions'],'response_tokens':v['response_tokens']})
    prefix='local_' if local else ''
    for name,rows in [('training_curve.csv',curve),('curriculum_coverage.csv',coverage),('evaluation_curve.csv',evaluation)]:
        with (output/(prefix+name)).open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(output/(prefix+'curriculum_audit.json'),result)
    atomic_json(output/(prefix+'comparison_summary.json'),{k:result[k] for k in (
        'summaries','paired','targeted_evaluation','targeted_paired','development_selection')})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local',action='store_true')
    a=p.parse_args();audit(a.output,a.local)

