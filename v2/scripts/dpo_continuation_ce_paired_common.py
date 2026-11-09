"""Frozen E4 data substitution: same DPO, CE coefficient and action budget."""
import sys,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from dpo_uniform_common import load,jsonl,tasks,contexts,seed_for
from dpo_chosen_ce_common import runtime_identity
from paired_runtime import check_job_runtime as check_runtime,runtime_report
from apt_v2.dpo_recovery_training import validate_training_pairs
from apt_v2.dpo_recovery import SCENARIOS,encode_recovery_pair
from apt_v2.hard_recovery_sft import recovery_weights
from apt_v2.dpo_investigation import build_probes
from apt_v2.recovery_diagnostics import messages_id,summarize
from apt_v2.sft_diagnostics import summarize_episodes
from apt_v2.sft import encode_example
from apt_v2.dpo_continuation_ce import build_continuations,ce_schedule,coverage,token_budget
CONFIG='dpo-continuation-ce-paired-v1.json'
VARIANTS=('chosen_control','continuation_ce')
def training_path(output,branch):
    if branch not in VARIANTS:raise ValueError('fixed E4 training branch required')
    return output/'training'/branch
def checkpoint_path(spec):return ROOT/'artifacts'/spec['run']/spec['checkpoint']
def checkpoints(cfg,output):return {b:(training_path(output,b)/'final' if b in VARIANTS else checkpoint_path(cfg['frozen_models'][b])) for b in cfg['evaluation_branches']}
def validate_frozen_model(spec):
    p=checkpoint_path(spec);c=read(p/'complete.json')
    if file_hash(p/'complete.json')!=spec['complete_sha256'] or file_hash(p/'adapter/adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('pinned frozen checkpoint differs')
    verify_files(p/'adapter',c['adapter_sha256'])
    return {'complete.json':file_hash(p/'complete.json'),**{'adapter/'+n:h for n,h in c['adapter_sha256'].items()}}
def check_contract(cfg):
    old=read(ROOT/'artifacts'/cfg['comparison_source_run']/'config.json')
    for k in ('initial_run','adapter_sha256','data_run','pairs_sha256','pair_count','max_length','hard_weight','epochs','max_steps','gradient_accumulation','learning_rate','max_grad_norm','seed','checkpoint_every','checkpoint_steps','source_sft_updates','source_dpo_updates','beta','eval','sample_eval','recovery_eval','task_counts','checkpoint_selection','reference_scores_sha256'):
        if cfg[k]!=old[k]:raise ValueError('fixed E3 DPO configuration differs: '+k)
    if cfg['chosen_ce_coefficient']!=old['training_variants']['dpo_ce']['chosen_ce_coefficient'] or cfg['ce_action_exposures']!=169*6 or cfg['ce_pool_actions']!=288 or cfg['maximum_optimizer_updates_this_job']!=516 or cfg['training_variants']!={'chosen_control':{'chosen_ce_coefficient':1},'continuation_ce':{'chosen_ce_coefficient':1}} or cfg['evaluation_branches']!=['weighted_sft','historical_dpo_ce','chosen_control','continuation_ce']:raise ValueError('fixed E4 CE substitution required')
    origin=ROOT/'artifacts'/cfg['comparison_source_run']
    for name,key in [('config.json','comparison_config_sha256'),('audit.json','comparison_audit_sha256'),('run_status.json','comparison_run_status_sha256')]:
        if file_hash(origin/name)!=cfg[key]:raise ValueError('reviewed E3 source differs: '+name)
    if read(origin/'execution_environment.json')!=cfg['required_execution_environment'] or read(origin/'run_status.json')['error'] is not None or not read(origin/'audit.json')['passed']:raise ValueError('reviewed E3 comparison required')
def validate_ce_inputs(output,cfg):
    examples,episodes=build_continuations();schedule=ce_schedule(len(examples),cfg['ce_action_exposures'],cfg['ce_seed'])
    if examples!=load(output/'ce_examples.jsonl') or episodes!=load(output/'ce_episodes.jsonl') or schedule!=read(output/'ce_schedule.json')['indices']:raise ValueError('expert CE data or deterministic schedule differs')
    if read(output/'ce_schedule.json')['seed']!=cfg['ce_seed'] or read(output/'ce_schedule.json')['pool_sha256']!=file_hash(output/'ce_examples.jsonl'):raise ValueError('CE sampler pool binding differs')
    if coverage(examples,schedule)!=read(output/'ce_summary.json')['coverage'] or len(examples)!=cfg['ce_pool_actions']:raise ValueError('expert CE coverage differs')
    dev={messages_id(p['messages']) for p in load(output/'probes.jsonl') if p['split']=='development'}
    if any(e['split']!='train' or messages_id(e['messages']) in dev for e in examples):raise ValueError('train-only CE context required')
    return examples,schedule
def verify_preflight(output):
    cfg=read(output/'config.json');p=read(output/'preflight.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed E4 config required')
    check_contract(cfg);verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,p['provenance'])
    for n,h in p['input_files_sha256'].items():
        if file_hash(output/n)!=h:raise ValueError('frozen E4 input differs: '+n)
    if file_hash(output/'reference_scores.json')!=cfg['reference_scores_sha256']:raise ValueError('byte-identical canonical reference required')
    for b,spec in cfg['frozen_models'].items():
        if validate_frozen_model(spec)!=p['checkpoints'][b]:raise ValueError('frozen model binding differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',p['model_files']);check_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'),precision=False)
    if runtime_report(cfg,read(output/'execution_environment.json'))!=read(output/'runtime_precheck.json') or not read(output/'runtime_precheck.json')['passed'] or contexts()!=load(output/'recovery_contexts.jsonl') or build_probes()!=load(output/'probes.jsonl') or read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('fixed runtime or development/probe schedule differs')
    validate_ce_inputs(output,cfg);return cfg

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed E4 config required')
    runtime=runtime_identity();write(output/'execution_environment.json',runtime);write(output/'runtime_precheck.json',runtime_report(cfg,runtime))
    check_runtime(cfg,runtime);check_contract(cfg)
    origin=ROOT/'artifacts'/cfg['reference_run'];comparison=ROOT/'artifacts'/cfg['comparison_source_run'];data=ROOT/'artifacts'/cfg['data_run'];provenance=[]
    for name,key in [('reference_scores.json','reference_scores_sha256'),('audit.json','reference_audit_sha256'),('audit_repair_status.json','reference_repair_status_sha256')]:
        if file_hash(origin/name)!=cfg[key]:raise ValueError('reviewed repaired E2 differs: '+name)
    if not read(origin/'audit.json')['passed'] or read(origin/'audit_repair_status.json')['error'] is not None or not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed reference and training pairs required')
    pairs=load(data/'pairs.jsonl');validate_training_pairs(pairs);weights,ws=recovery_weights(pairs,4)
    frozen={b:validate_frozen_model(s) for b,s in cfg['frozen_models'].items()};identity=read(ROOT/'artifacts'/cfg['initial_run']/'model_identity.json')
    for spec in cfg['frozen_models'].values():
        run=ROOT/'artifacts'/spec['run'];checkpoint=checkpoint_path(spec)
        if not read(run/'audit.json')['passed'] or read(run/'model_identity.json')!=identity:raise ValueError('audited shared base identity required')
        provenance.extend(run/n for n in ('config.json','audit.json','model_identity.json'));provenance.extend([checkpoint/'complete.json',checkpoint.parent/'training_summary.json'])
    if identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('pinned base revision differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',identity['files_sha256'])
    for name,source in [('pairs.jsonl',data/'pairs.jsonl'),('reference_scores.json',origin/'reference_scores.json')]:shutil.copyfile(source,output/name)
    write(output/'weights.json',{'weights':weights,'summary':ws});write(output/'model_identity.json',identity)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'.cache/qwen2.5-1.5b-instruct'/n,p)
    probes=build_probes();cs=contexts()
    if probes!=load(comparison/'probes.jsonl') or cs!=load(comparison/'recovery_contexts.jsonl'):raise ValueError('reviewed E3 expert contexts differ')
    jsonl(output/'probes.jsonl',probes);jsonl(output/'recovery_contexts.jsonl',cs);write(output/'development_tasks.json',[asdict(t) for t in tasks()])
    examples,episodes=build_continuations();schedule=ce_schedule(len(examples),cfg['ce_action_exposures'],cfg['ce_seed'])
    jsonl(output/'ce_examples.jsonl',examples);jsonl(output/'ce_episodes.jsonl',episodes);write(output/'ce_schedule.json',{'seed':cfg['ce_seed'],'indices':schedule,'pool_sha256':file_hash(output/'ce_examples.jsonl')})
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ce=[encode_example(tok,e,cfg['max_length']) for e in examples]
    write(output/'ce_summary.json',{'coverage':coverage(examples,schedule),'token_budget':token_budget(encoded,ce,schedule,cfg['epochs']),'maximum_CE_input_tokens':max(len(e['input_ids']) for e in ce),'sampler':'shuffled full-pool cycles without replacement; truncate final cycle','coefficient':1,'CE_loss':'uniform action mean of response-token mean including EOS','DPO_part':'unchanged from E3','comparison_E3_CE_response_tokens':read(comparison/'training/dpo_ce/training_summary.json')['chosen_response_tokens_per_epoch']*cfg['epochs'],'comparison_E3_CE_extra_forwards':0})
    write(output/'control_ce_summary.json',{'source':'original169_chosen','CE_action_exposures':1014,'CE_response_tokens':cfg['epochs']*sum(len(e['chosen']) for e in encoded),'CE_extra_forward_input_tokens':0,'CE_extra_forwards':0,'DPO_pair_exposures':1014,'DPO_chosen_response_tokens':cfg['epochs']*sum(len(e['chosen']) for e in encoded),'DPO_rejected_response_tokens':cfg['epochs']*sum(len(e['rejected']) for e in encoded),'DPO_input_tokens':cfg['epochs']*sum(2*len(e['prompt'])+len(e['chosen'])+len(e['rejected']) for e in encoded)})
    provenance.extend(origin/n for n in ('reference_scores.json','audit.json','audit_repair_status.json','reference_numerics_audit.json'))
    provenance.extend(data/n for n in ('pairs.jsonl','audit.json','pair_summary.json'));provenance.extend(comparison/n for n in ('run_status.json','execution_environment.json','probes.jsonl','recovery_contexts.jsonl','reference_scores.json'))
    for p in set(provenance):
        target=output/'provenance'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    names=('config.json','pairs.jsonl','weights.json','reference_scores.json','model_identity.json','probes.jsonl','recovery_contexts.jsonl','development_tasks.json','execution_environment.json','runtime_precheck.json','control_ce_summary.json','ce_examples.jsonl','ce_episodes.jsonl','ce_schedule.json','ce_summary.json')
    write(output/'preflight.json',{'input_files_sha256':{n:file_hash(output/n) for n in names},'checkpoints':frozen,'model_files':identity['files_sha256'],'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(provenance))},'pairs':169,'probes':238,'contexts':28,'training_variants':cfg['training_variants'],'test_split_used':False})
    verify_preflight(output)

def verify_inputs(output):
    cfg=verify_preflight(output)
    for b in VARIANTS:
        child=training_path(output,b);binding=read(child/'input_binding.json')
        if binding['root_input_files_sha256']!=read(output/'preflight.json')['input_files_sha256'] or binding['branch']!=b or binding['chosen_ce_coefficient']!=cfg['training_variants'][b]['chosen_ce_coefficient'] or binding['reference_scores_sha256']!=cfg['reference_scores_sha256']:raise ValueError('paired training binding differs')
        expected_algorithm='weighted_dpo_chosen_ce' if b=='chosen_control' else 'weighted_dpo_continuation_ce'
        if read(child/'config.json')!=dict(cfg,branch=b,chosen_ce_coefficient=cfg['training_variants'][b]['chosen_ce_coefficient'],branch_algorithm=expected_algorithm) or binding['config_sha256']!=file_hash(child/'config.json') or binding['weights_sha256']!=file_hash(child/'weights.json') or file_hash(child/'pairs.jsonl')!=cfg['pairs_sha256'] or file_hash(child/'reference_scores.json')!=cfg['reference_scores_sha256']:raise ValueError('actual branch training files differ')
        if read(child/'weights.json')!=read(output/'weights.json') or read(child/'model_identity.json')!=read(output/'model_identity.json') or binding['source_sha256']!=read(output/'run_manifest.json')['source_sha256']:raise ValueError('branch source/data identity differs')
        verify_files(ROOT,binding['provenance'])
        for n,h in read(output/'model_identity.json')['files_sha256'].items():
            if n.endswith(('.json','.txt','.jinja')) and file_hash(child/'tokenizer'/n)!=h:raise ValueError('branch tokenizer differs')
        for name in (*[f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']],'final'):
            p=child/name;verify_files(p/'adapter',read(p/'complete.json')['adapter_sha256'])
            if read(p/'binding.json')!=binding:raise ValueError('paired checkpoint binding differs')
    return cfg


def model_specs(cfg,output):
    result=dict(cfg['frozen_models'])
    for b in VARIANTS:
        p=training_path(output,b)/'final';result[b]={'run':cfg['run_id'],'checkpoint':p.relative_to(output).as_posix(),'adapter_sha256':file_hash(p/'adapter/adapter_model.safetensors'),'complete_sha256':file_hash(p/'complete.json')}
    return result

def result_summary(complete,recoveries,actions,cfg):
    result={}
    def paired(a,b,key):
        x={key(e):e['success'] for e in a};y={key(e):e['success'] for e in b}
        if len(x)!=len(a) or len(y)!=len(b) or x.keys()!=y.keys():raise ValueError('distinct paired evaluation keys required')
        return {'both_success':sum(x[k] and y[k] for k in x),'both_failure':sum(not x[k] and not y[k] for k in x),
            'improved':sum(not x[k] and y[k] for k in x),'regressed':sum(x[k] and not y[k] for k in x)}
    def es(es):return {'episodes':len(es),'successes':sum(e['success'] for e in es),'turns':sum(len(e['turns']) for e in es),'tool_errors':sum(e['tool_errors'] for e in es)}
    for b in cfg['evaluation_branches']:
        result[b]={}
        for mode in ('greedy','sample'):
            full=[e for e in complete if e['model_branch']==b and e['mode']==mode];rs=[e for e in recoveries if e['model_branch']==b and e['mode']==mode];aa=[r for r in actions if r['model_branch']==b and r['mode']==mode]
            result[b][mode]={'complete':{**summarize_episodes(full),'episodes':len(full)},'complete_by_task':{t.id:es([e for e in full if e['task_id']==t.id]) for t in tasks()},
                'recovery':es(rs),'recovery_by_task':{t.id:es([e for e in rs if e['task_id']==t.id]) for t in tasks()},
                'recovery_by_scenario':{s:es([e for e in rs if e['scenario']==s]) for s in SCENARIOS},'first_action':summarize(aa)}
    paired_results={}
    for baseline,target in [('chosen_control','continuation_ce'),('weighted_sft','continuation_ce'),('historical_dpo_ce','chosen_control')]:
        paired_results[baseline+'_to_'+target]={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_'+target][mode]={'complete':paired(pick(complete,baseline),pick(complete,target),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,target),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
