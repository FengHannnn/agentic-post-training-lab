"""One weighted-SFT start, a fresh matching reference, bounded short DPO."""
import sys,shutil,copy
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from dpo_uniform_common import load,jsonl,tasks,contexts,seed_for
from dpo_chosen_ce_common import runtime_identity
from paired_runtime import check_job_runtime,runtime_report
from apt_v2.dpo_recovery_training import validate_training_pairs,verify_reference
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.recovery_diagnostics import messages_id
CONFIG='dpo-sft-start-short-v1.json'

def branch_name(step):return f'dpo_{step:04}'
def checkpoint(output,step):return output/'training'/f'checkpoint-{step:04}'
def normalize_config(c):
    c=copy.deepcopy(c)
    if isinstance(c.get('target_modules'),list):c['target_modules']=sorted(c['target_modules'])
    return c

def verify_sources(cfg):
    fixed={'algorithm':'uniform_dpo_from_weighted_sft','initial_run':'hard-recovery-sft-weighted-v1-20261103','adapter_sha256':'f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f','pair_count':169,'source_sft_updates':513,'source_dpo_updates':11,'gradient_accumulation':4,'hard_weight':1,'learning_rate':5e-7,'beta':.1,'checkpoint_steps':[4,12,24],'max_steps':24}
    if any(cfg[k]!=v for k,v in fixed.items()):raise ValueError('fixed short DPO protocol differs')
    if cfg['evaluation_branches']!=['weighted_sft']+[branch_name(s) for s in cfg['checkpoint_steps']]:raise ValueError('fixed ascending candidate schedule required')
    source=ROOT/'artifacts'/cfg['source_comparison_run'];parent=ROOT/'artifacts'/cfg['initial_run'];data=ROOT/'artifacts'/cfg['data_run']
    for n,k in [('config.json','source_comparison_config_sha256'),('audit.json','source_comparison_audit_sha256'),('run_status.json','source_comparison_status_sha256')]:
        if file_hash(source/n)!=cfg[k]:raise ValueError('reviewed direct comparison differs: '+n)
    if not read(source/'audit.json')['passed'] or read(source/'run_status.json')['error'] is not None:raise ValueError('audited direct comparison required')
    previous=read(source/'config.json')
    for k in ('eval','sample_eval','recovery_eval','protocol'):
        if cfg[k]!=previous[k]:raise ValueError('fixed direct comparison evaluation differs: '+k)
    if file_hash(parent/'audit.json')!=cfg['initial_audit_sha256'] or not read(parent/'audit.json')['passed']:raise ValueError('audited weighted SFT required')
    if file_hash(parent/'final/complete.json')!=cfg['complete_sha256']:raise ValueError('weighted SFT completion differs')
    complete=read(parent/'final/complete.json');verify_files(parent/'final/adapter',complete['adapter_sha256'])
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256'] or complete['total_sft_updates']!=513 or complete['total_dpo_updates']!=11:raise ValueError('SFT513/DPO11 weighted SFT required')
    if file_hash(data/'audit.json')!=cfg['data_audit_sha256'] or not read(data/'audit.json')['passed'] or file_hash(data/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('audited preference data differs')
    return source,parent,data

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed short DPO config required')
    actual=runtime_identity();write(output/'execution_environment.json',actual);write(output/'runtime_precheck.json',runtime_report(cfg,actual));check_job_runtime(cfg,actual)
    source,parent,data=verify_sources(cfg);identity=read(parent/'model_identity.json')
    if identity!=read(source/'model_identity.json'):raise ValueError('reviewed base model differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',identity['files_sha256']);write(output/'model_identity.json',identity)
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'.cache/qwen2.5-1.5b-instruct'/n,p)
    pairs=load(data/'pairs.jsonl');validate_training_pairs(pairs);cs=contexts()
    if any(messages_id(c['messages']) in {messages_id(p['messages']) for p in pairs} for c in cs):raise ValueError('development/train prompt overlap')
    if cs!=load(source/'recovery_contexts.jsonl') or read(source/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('fixed development schedule differs')
    shutil.copyfile(data/'pairs.jsonl',output/'pairs.jsonl');jsonl(output/'recovery_contexts.jsonl',cs);write(output/'development_tasks.json',[asdict(t) for t in tasks()]);write(output/'weights.json',{'weights':[1.]*169,'hard_weight':1,'reduction':'mean(sequence-sum DPO over pairs)'})
    adapter=output/'adapters/weighted_sft';adapter.mkdir(parents=True);shutil.copyfile(parent/'final/adapter/adapter_model.safetensors',adapter/'adapter_model.safetensors');write(adapter/'adapter_config.json',normalize_config(read(parent/'final/adapter/adapter_config.json')))
    originals=output/'source_checkpoint';originals.mkdir();shutil.copyfile(parent/'final/complete.json',originals/'complete.json');shutil.copyfile(parent/'final/adapter/adapter_config.json',originals/'adapter_config.json')
    provenance=[source/n for n in ('config.json','audit.json','run_status.json','model_identity.json')]+[parent/n for n in ('config.json','audit.json','final/complete.json','final/adapter/adapter_config.json','model_identity.json')]+[data/n for n in ('audit.json','pairs.jsonl')]
    for p in provenance:
        saved=output/'provenance'/p.relative_to(ROOT);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,saved)
    names=['config.json','execution_environment.json','runtime_precheck.json','model_identity.json','pairs.jsonl','weights.json','recovery_contexts.jsonl','development_tasks.json','source_checkpoint/complete.json','source_checkpoint/adapter_config.json','adapters/weighted_sft/adapter_config.json','adapters/weighted_sft/adapter_model.safetensors']
    write(output/'input_binding.json',{'input_files_sha256':{n:file_hash(output/n) for n in names},'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance},'initial_and_reference_adapter_sha256':cfg['adapter_sha256'],'maximum_optimizer_updates':24,'initial_optimizer_updates':0})
    verify_inputs(output)

def verify_inputs(output,precision=False):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs'/CONFIG):raise ValueError('fixed short DPO config differs')
    verify_sources(cfg);verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(output,binding['input_files_sha256']);verify_files(ROOT,binding['provenance'])
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',read(output/'model_identity.json')['files_sha256'])
    check_job_runtime(cfg,runtime_identity(),read(output/'execution_environment.json'),precision=precision)
    if runtime_report(cfg,read(output/'execution_environment.json'))!=read(output/'runtime_precheck.json') or not read(output/'runtime_precheck.json')['passed']:raise ValueError('runtime precheck differs')
    if load(output/'recovery_contexts.jsonl')!=contexts() or read(output/'development_tasks.json')!=[asdict(t) for t in tasks()]:raise ValueError('development schedule differs')
    if read(output/'weights.json')!={'weights':[1.]*169,'hard_weight':1,'reduction':'mean(sequence-sum DPO over pairs)'}:raise ValueError('uniform pair weights required')
    if (output/'reference_binding.json').exists():
        ref=read(output/'reference_binding.json')
        if ref!={'reference_scores_sha256':file_hash(output/'reference_scores.json'),'baseline_adapter_sha256':cfg['adapter_sha256'],'pairs_sha256':cfg['pairs_sha256'],'before_optimizer_update':0,'dtype':'float32','mode':'eval'}:raise ValueError('fresh immutable weighted SFT reference differs')
    return cfg

def verify_reference_cache(output,cfg,tok):
    pairs=load(output/'pairs.jsonl');validate_training_pairs(pairs);encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
    verify_reference(pairs,encoded,read(output/'reference_scores.json'),cfg['adapter_sha256'])
    if read(output/'reference_binding.json')['reference_scores_sha256']!=file_hash(output/'reference_scores.json'):raise ValueError('reference cache mutated')
    return encoded

from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.recovery_diagnostics import summarize
from apt_v2.sft_diagnostics import summarize_episodes

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
    for baseline,target in [('weighted_sft',b) for b in cfg['evaluation_branches'] if b!='weighted_sft' and 'weighted_sft' in cfg['evaluation_branches']]:
        paired_results[baseline+'_to_'+target]={}
        for mode in ('greedy','sample'):
            pick=lambda rows,b:[e for e in rows if e['model_branch']==b and e['mode']==mode]
            paired_results[baseline+'_to_'+target][mode]={'complete':paired(pick(complete,baseline),pick(complete,target),lambda e:(e['task_id'],e['rollout_index'])),
                'recovery':paired(pick(recoveries,baseline),pick(recoveries,target),lambda e:(e['context_id'],e['sample_index']))}
    return {'branches':result,'paired':paired_results,'test_split_used':False,'recovery_turn_budget':'original task max_turns includes injected prefix'}




if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
