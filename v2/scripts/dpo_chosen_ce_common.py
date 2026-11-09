"""Frozen E3 pairing, exact cached reference, and same-runtime guards."""
import sys,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from dpo_uniform_common import load,jsonl,tasks,contexts,seed_for
from apt_v2.dpo_recovery_training import validate_training_pairs
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.hard_recovery_sft import recovery_weights
from apt_v2.dpo_investigation import build_probes
from apt_v2.recovery_diagnostics import messages_id,summarize
from apt_v2.sft_diagnostics import summarize_episodes

CONFIG='dpo-chosen-ce-control-v1.json'
VARIANTS=('weighted_control','dpo_ce')

def runtime_identity():
    import torch,platform,importlib.metadata
    if not torch.cuda.is_available():raise ValueError('CUDA required for pinned E3 execution')
    p=torch.cuda.get_device_properties(0)
    return {'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':str(getattr(p,'uuid',None)),'compute_capability':list(torch.cuda.get_device_capability(0)),'gpu_total_memory':p.total_memory,'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,'packages':{n:importlib.metadata.version(n) for n in ('transformers','peft','vllm')}}

def check_runtime(cfg,actual,expected=None,precision=True):
    if actual['gpu_name']!=cfg['required_gpu_name']:raise ValueError('fixed NVIDIA L40S hardware required before updates')
    if expected is not None:
        keys=set(expected) if precision else set(expected)-{'float32_matmul_precision','allow_tf32_matmul','allow_tf32_cudnn'}
        if set(actual)!=set(expected) or any(actual[k]!=expected[k] for k in keys):raise ValueError('paired runtime identity differs')

def training_path(output,branch):
    if branch not in VARIANTS:raise ValueError('fixed training branch required')
    return output/'training'/branch

def checkpoint_path(spec):return ROOT/'artifacts'/spec['run']/spec['checkpoint']
def checkpoints(cfg,output):
    return {'weighted_sft':checkpoint_path(cfg['frozen_models']['weighted_sft']),'uniform_dpo':checkpoint_path(cfg['frozen_models']['uniform_dpo']),**{b:training_path(output,b)/'final' for b in VARIANTS}}

def check_contract(cfg):
    old=read(ROOT/'artifacts/hard-recovery-dpo-matched-v1-20261103/config.json')
    for k in ('initial_run','adapter_sha256','data_run','pairs_sha256','pair_count','max_length','hard_weight','epochs','max_steps','gradient_accumulation','learning_rate','max_grad_norm','seed','checkpoint_every','checkpoint_steps','source_sft_updates','source_dpo_updates','beta','eval','sample_eval','recovery_eval','task_counts','checkpoint_selection'):
        if cfg[k]!=old[k]:raise ValueError('fixed original weighted configuration differs: '+k)
    if cfg['training_variants']!={'weighted_control':{'chosen_ce_coefficient':0},'dpo_ce':{'chosen_ce_coefficient':1}} or cfg['evaluation_branches']!=['weighted_sft','uniform_dpo','weighted_control','dpo_ce'] or cfg['maximum_optimizer_updates_this_job']!=516:raise ValueError('fixed paired CE experiment required')

def validate_frozen_model(spec):
    p=checkpoint_path(spec);c=read(p/'complete.json')
    if file_hash(p/'complete.json')!=spec['complete_sha256'] or file_hash(p/'adapter/adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('pinned frozen checkpoint differs')
    verify_files(p/'adapter',c['adapter_sha256'])
    return {'complete.json':file_hash(p/'complete.json'),**{'adapter/'+n:h for n,h in c['adapter_sha256'].items()}}

def verify_preflight(output):
    cfg=read(output/'config.json');p=read(output/'preflight.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed E3 config required')
    check_contract(cfg);verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,p['provenance'])
    for n,h in p['input_files_sha256'].items():
        if file_hash(output/n)!=h:raise ValueError('frozen E3 input differs: '+n)
    if file_hash(output/'reference_scores.json')!=cfg['reference_scores_sha256']:raise ValueError('byte-identical canonical reference required')
    for b,spec in cfg['frozen_models'].items():
        if validate_frozen_model(spec)!=p['checkpoints'][b]:raise ValueError('frozen model binding differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',p['model_files'])
    check_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'),precision=False)
    if contexts()!=load(output/'recovery_contexts.jsonl') or build_probes()!=load(output/'probes.jsonl') or read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('fixed development/probe schedule differs')
    return cfg

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed E3 config required')
    check_contract(cfg);runtime=runtime_identity();check_runtime(cfg,runtime);write(output/'execution_environment.json',runtime)
    origin=ROOT/'artifacts'/cfg['reference_run'];data=ROOT/'artifacts'/cfg['data_run'];provenance=[]
    for name,key in [('reference_scores.json','reference_scores_sha256'),('audit.json','reference_audit_sha256'),('audit_repair_status.json','reference_repair_status_sha256')]:
        if file_hash(origin/name)!=cfg[key]:raise ValueError('reviewed repaired E2 differs: '+name)
    rs=read(origin/'audit_repair_status.json')
    if not read(origin/'audit.json')['passed'] or rs['error'] is not None or rs['exit_code']!=0 or not rs['original_artifacts_preserved']:raise ValueError('completed repaired E2 audit required')
    if not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed training pairs required')
    pairs=load(data/'pairs.jsonl');validate_training_pairs(pairs);weights,ws=recovery_weights(pairs,4)
    frozen={b:validate_frozen_model(s) for b,s in cfg['frozen_models'].items()};identity=read(ROOT/'artifacts'/cfg['initial_run']/'model_identity.json')
    for spec in cfg['frozen_models'].values():
        run=ROOT/'artifacts'/spec['run']
        if not read(run/'audit.json')['passed'] or read(run/'model_identity.json')!=identity:raise ValueError('audited shared base identity required')
        provenance.extend(run/n for n in ('config.json','audit.json','model_identity.json','training_summary.json','final/complete.json'))
    if identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('pinned base revision differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',identity['files_sha256'])
    for name,source in [('pairs.jsonl',data/'pairs.jsonl'),('reference_scores.json',origin/'reference_scores.json')]:shutil.copyfile(source,output/name)
    write(output/'weights.json',{'weights':weights,'summary':ws});write(output/'model_identity.json',identity)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'.cache/qwen2.5-1.5b-instruct'/n,p)
    probes=build_probes();cs=contexts()
    if probes!=load(origin/'probes.jsonl') or cs!=load(origin/'recovery_contexts.jsonl'):raise ValueError('reviewed E2 expert contexts differ')
    if any(messages_id(c['messages']) in {messages_id(p['messages']) for p in pairs} for c in cs):raise ValueError('development context overlaps train')
    jsonl(output/'probes.jsonl',probes);jsonl(output/'recovery_contexts.jsonl',cs);write(output/'development_tasks.json',[asdict(t) for t in tasks()])
    provenance.extend(origin/n for n in ('reference_scores.json','audit_repair_status.json','reference_numerics_audit.json','config.json','probes.jsonl','recovery_contexts.jsonl'))
    provenance.extend(data/n for n in ('pairs.jsonl','audit.json','pair_summary.json'));provenance.append(ROOT/'artifacts/hard-recovery-dpo-matched-v1-20261103/config.json')
    for p in set(provenance):
        target=output/'provenance'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    names=('config.json','pairs.jsonl','weights.json','reference_scores.json','model_identity.json','probes.jsonl','recovery_contexts.jsonl','development_tasks.json','execution_environment.json')
    write(output/'preflight.json',{'input_files_sha256':{n:file_hash(output/n) for n in names},'checkpoints':frozen,'model_files':identity['files_sha256'],'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(provenance))},'pairs':169,'probes':238,'contexts':28,'training_variants':cfg['training_variants'],'test_split_used':False})
    verify_preflight(output)

def verify_inputs(output):
    cfg=verify_preflight(output)
    for b in VARIANTS:
        child=training_path(output,b);binding=read(child/'input_binding.json')
        if binding['root_input_files_sha256']!=read(output/'preflight.json')['input_files_sha256'] or binding['branch']!=b or binding['chosen_ce_coefficient']!=cfg['training_variants'][b]['chosen_ce_coefficient'] or binding['reference_scores_sha256']!=cfg['reference_scores_sha256']:raise ValueError('paired training binding differs')
        expected_algorithm='hard_recovery_weighted_dpo' if cfg['training_variants'][b]['chosen_ce_coefficient']==0 else 'weighted_dpo_chosen_ce'
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
    for baseline,target in [('weighted_control','dpo_ce'),('uniform_dpo','weighted_control'),('weighted_sft','dpo_ce')]:
        paired_results[baseline+'_to_'+target]={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_'+target][mode]={'complete':paired(pick(complete,baseline),pick(complete,target),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,target),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
