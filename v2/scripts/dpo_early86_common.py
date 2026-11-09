"""Frozen early DPO86 versus weighted SFT inputs and direct paired results."""
import sys,shutil,json,copy
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from dpo_uniform_common import load,jsonl,tasks,contexts,seed_for
from dpo_chosen_ce_common import runtime_identity
from paired_runtime import check_job_runtime,runtime_report
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.recovery_diagnostics import summarize
from apt_v2.sft_diagnostics import summarize_episodes
CONFIG='dpo-early86-vs-weighted-sft-v1.json'
def checkpoint_path(spec):return ROOT/'artifacts'/spec['run']/spec['checkpoint']
def normalize_config(config):
    result=copy.deepcopy(config)
    if isinstance(result.get('target_modules'),list):result['target_modules']=sorted(result['target_modules'])
    return result

def model_binding(spec):
    p=checkpoint_path(spec);c=read(p/'complete.json')
    if file_hash(p/'complete.json')!=spec['complete_sha256'] or file_hash(p/'adapter/adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('frozen inference model differs')
    verify_files(p/'adapter',c['adapter_sha256'])
    return {'complete.json':spec['complete_sha256'],**{'adapter/'+n:h for n,h in c['adapter_sha256'].items()}}
def evaluation_runtime_report(cfg,actual):
    result=runtime_report(cfg,actual);result['scope']='same-job_inference_pair';return result

def verify_source(cfg):
    source=ROOT/'artifacts'/cfg['source_run']
    for n,k in [('config.json','source_config_sha256'),('audit.json','source_audit_sha256'),('run_status.json','source_run_status_sha256')]:
        if file_hash(source/n)!=cfg[k]:raise ValueError('reviewed E1 source differs: '+n)
    if not read(source/'audit.json')['passed'] or read(source/'run_status.json')['error'] is not None:raise ValueError('completed audited E1 required')
    if cfg['training_enabled'] is not False or cfg['optimizer_updates']!=0 or cfg['protocol']!={'one_model_per_process':True,'max_loras':1,'lora_int_id':1,'lora_name':'fixed_adapter','normalize_target_modules_order':True,'quality_contexts_per_model':7,'quality_repeats':2}:raise ValueError('fixed two-model inference protocol required')
    if cfg['evaluation_branches']!=['weighted_sft','dpo_0086'] or set(cfg['frozen_models'])!={'weighted_sft','dpo_0086'}:raise ValueError('fixed SFT and early DPO86 comparison required')
    original=read(source/'config.json')
    for branch,spec in cfg['frozen_models'].items():
        if any(spec[k]!=original['checkpoints'][branch][k] for k in ('run','checkpoint','adapter_sha256','complete_sha256')):raise ValueError('frozen E1 candidate differs')
    for k in ('eval','sample_eval'):
        if cfg[k]!=original[k]:raise ValueError('fixed E1 full-task evaluation schedule differs')
    if cfg['recovery_eval']!={'samples_per_context':3,'temperature':.8,'seed':2026110303,'complete_continuation':True}:raise ValueError('fixed complete-recovery evaluation required')
    origin=ROOT/'artifacts'/cfg['origin_run']
    if file_hash(origin/'config.json')!=cfg['origin_config_sha256'] or file_hash(origin/'input_binding.json')!=cfg['origin_binding_sha256']:raise ValueError('matched DPO origin binding differs')
    for b,spec in cfg['frozen_models'].items():
        c=read(checkpoint_path(spec)/'complete.json')
        if file_hash(checkpoint_path(spec)/'complete.json')!=spec['complete_sha256']:raise ValueError('fixed checkpoint metadata differs')
        if b=='dpo_0086' and (c['algorithm']!='hard_recovery_weighted_dpo' or c['steps']!=86 or c['epochs_completed']!=2 or c['cursor']!=338 or c['total_sft_updates']!=255 or c['total_dpo_updates']!=97):raise ValueError('exact early 86-step ancestry required')
        if b=='weighted_sft' and (c['steps']!=258 or c['total_sft_updates']!=513 or c['total_dpo_updates']!=11):raise ValueError('fixed weighted SFT ancestry required')

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed confirmation config required')
    actual=runtime_identity();write(output/'execution_environment.json',actual);write(output/'runtime_precheck.json',evaluation_runtime_report(cfg,actual));check_job_runtime(cfg,actual);verify_source(cfg)
    source=ROOT/'artifacts'/cfg['source_run'];identity=read(source/'model_identity.json');verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',identity['files_sha256']);write(output/'model_identity.json',identity)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            target=output/'tokenizer'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'.cache/qwen2.5-1.5b-instruct'/n,target)
    cs=contexts()
    if cs!=load(source/'recovery_contexts.jsonl') or read(source/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('reviewed development schedule differs')
    jsonl(output/'recovery_contexts.jsonl',cs);write(output/'development_tasks.json',[asdict(t) for t in tasks()]);bindings={};normalized={}
    for branch,spec in cfg['frozen_models'].items():
        bindings[branch]=model_binding(spec);original=checkpoint_path(spec)/'adapter';target=output/'adapters'/branch;target.mkdir(parents=True)
        shutil.copyfile(original/'adapter_model.safetensors',target/'adapter_model.safetensors');config=normalize_config(read(original/'adapter_config.json'));write(target/'adapter_config.json',config)
        saved=output/'source_checkpoints'/branch;saved.mkdir(parents=True);shutil.copyfile(checkpoint_path(spec)/'complete.json',saved/'complete.json');shutil.copyfile(original/'adapter_config.json',saved/'adapter_config.json')
        normalized[branch]={'original_config_sha256':file_hash(original/'adapter_config.json'),'normalized_config_sha256':file_hash(target/'adapter_config.json'),'adapter_sha256':file_hash(target/'adapter_model.safetensors')}
    names=('config.json','model_identity.json','execution_environment.json','runtime_precheck.json','recovery_contexts.jsonl','development_tasks.json')
    write(output/'input_binding.json',{'input_files_sha256':{n:file_hash(output/n) for n in names},'model_files':identity['files_sha256'],'original_checkpoints':bindings,'normalized_adapters':normalized,'training_enabled':False,'optimizer_updates':0});verify_inputs(output)

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed confirmation config required')
    verify_source(cfg);verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    for n,h in binding['input_files_sha256'].items():
        if file_hash(output/n)!=h:raise ValueError('frozen confirmation input differs: '+n)
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files']);check_job_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'),precision=False)
    if evaluation_runtime_report(cfg,read(output/'execution_environment.json'))!=read(output/'runtime_precheck.json') or not read(output/'runtime_precheck.json')['passed']:raise ValueError('runtime precheck differs')
    for b,spec in cfg['frozen_models'].items():
        if model_binding(spec)!=binding['original_checkpoints'][b]:raise ValueError('source model binding differs')
        target=output/'adapters'/b;original=checkpoint_path(spec)/'adapter';saved=binding['normalized_adapters'][b]
        if read(target/'adapter_config.json')!=normalize_config(read(original/'adapter_config.json')) or file_hash(target/'adapter_config.json')!=saved['normalized_config_sha256'] or file_hash(target/'adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('normalized inference adapter differs')
    if contexts()!=load(output/'recovery_contexts.jsonl') or read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('development schedule differs')
    return cfg

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
    for baseline,target in [('weighted_sft','dpo_0086')]:
        paired_results[baseline+'_to_'+target]={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_'+target][mode]={'complete':paired(pick(complete,baseline),pick(complete,target),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,target),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
