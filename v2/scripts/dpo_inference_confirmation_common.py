"""Inference-only confirmation with a repeated identical control and fixed LoRA position."""
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
CONFIG='dpo-inference-confirmation-v1.json'
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
def verify_source(cfg):
    source=ROOT/'artifacts'/cfg['source_run']
    for n,k in [('config.json','source_config_sha256'),('audit.json','source_audit_sha256'),('run_status.json','source_run_status_sha256')]:
        if file_hash(source/n)!=cfg[k]:raise ValueError('reviewed E4 source differs: '+n)
    if not read(source/'audit.json')['passed'] or read(source/'run_status.json')['error'] is not None:raise ValueError('completed audited E4 required')
    if cfg['training_enabled'] is not False or cfg['optimizer_updates']!=0 or cfg['protocol']!={'one_model_per_process':True,'max_loras':1,'lora_int_id':1,'lora_name':'fixed_adapter','normalize_target_modules_order':True,'duplicate_control_exact_responses_required':True}:raise ValueError('fixed inference-only protocol required')
    if cfg['frozen_models']['chosen_control']!=cfg['frozen_models']['control_repeat'] or cfg['evaluation_branches']!=['weighted_sft','chosen_control','control_repeat','continuation_ce']:raise ValueError('identical repeated control required')
    original=read(source/'config.json')
    for k in ('eval','sample_eval','recovery_eval'):
        if cfg[k]!=original[k]:raise ValueError('fixed E4 evaluation schedule differs')

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed confirmation config required')
    actual=runtime_identity();write(output/'execution_environment.json',actual);write(output/'runtime_precheck.json',runtime_report(cfg,actual));check_job_runtime(cfg,actual);verify_source(cfg)
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
    if runtime_report(cfg,read(output/'execution_environment.json'))!=read(output/'runtime_precheck.json') or not read(output/'runtime_precheck.json')['passed']:raise ValueError('runtime precheck differs')
    for b,spec in cfg['frozen_models'].items():
        if model_binding(spec)!=binding['original_checkpoints'][b]:raise ValueError('source model binding differs')
        target=output/'adapters'/b;original=checkpoint_path(spec)/'adapter';saved=binding['normalized_adapters'][b]
        if read(target/'adapter_config.json')!=normalize_config(read(original/'adapter_config.json')) or file_hash(target/'adapter_config.json')!=saved['normalized_config_sha256'] or file_hash(target/'adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('normalized inference adapter differs')
    if contexts()!=load(output/'recovery_contexts.jsonl') or read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('development schedule differs')
    return cfg

def repeatability(output):
    evidence={};maximum=0.
    for kind in ('complete','recovery'):
        key=lambda e:(e['task_id'] if kind=='complete' else e['context_id'],e['mode'],e['rollout_index'] if kind=='complete' else e['sample_index'])
        left=load(output/('chosen_control-'+kind+'-episodes.jsonl'));right=load(output/('control_repeat-'+kind+'-episodes.jsonl'));mapped={key(e):e for e in right}
        if len(mapped)!=len(right) or {key(e) for e in left}!=mapped.keys():raise ValueError('distinct repeat coverage required')
        mismatches=[]
        for e in left:
            a=copy.deepcopy(e);b=copy.deepcopy(mapped[key(e)]);a.pop('model_branch');b.pop('model_branch')
            for x,y in zip(a['turns'],b['turns']):
                lx=x['generation'].pop('raw_token_log_probs');ly=y['generation'].pop('raw_token_log_probs')
                if x['generation']['response_token_ids']==y['generation']['response_token_ids']:maximum=max(maximum,max((abs(v-w) for v,w in zip(lx,ly)),default=0.))
            if a!=b:mismatches.append(list(key(e)))
        evidence[kind]={'episodes':len(left),'identical':len(left)-len(mismatches),'mismatches':mismatches}
    return {'passed':all(not v['mismatches'] for v in evidence.values()),'branches':evidence,'maximum_raw_logp_difference_on_aligned_actions':maximum,'criterion':'identical complete recorded episodes excluding branch label and raw logp floats'}

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
    for baseline,target in [('chosen_control','continuation_ce'),('weighted_sft','continuation_ce'),('chosen_control','control_repeat')]:
        paired_results[baseline+'_to_'+target]={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_'+target][mode]={'complete':paired(pick(complete,baseline),pick(complete,target),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,target),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}



if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
