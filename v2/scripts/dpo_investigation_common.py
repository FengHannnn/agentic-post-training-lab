"""Frozen inputs and checkpoint registry for the inference-only E1 diagnosis."""
import sys,json,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.dpo_investigation import build_probes,investigation_tasks,investigation_contexts
from apt_v2.dpo_recovery_training import validate_training_pairs
from apt_v2.matched_dpo import verify_matched_updates
from apt_v2.hard_recovery_sft import recovery_weights
CONFIG='dpo-regression-checkpoints-v1.json'

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
def jsonl(path,rows):
    with path.open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def checkpoint_path(spec):return ROOT/'artifacts'/spec['run']/spec['checkpoint']
def seed_for(cfg,task,mode,ri,context=None):
    ti=[t.id for t in investigation_tasks()].index(task.id)
    if context is not None:
        from apt_v2.dpo_recovery import SCENARIOS
        return cfg['recovery_eval']['seed']+ti*100000+list(SCENARIOS).index(context['scenario'])*1000+ri*100
    return cfg['eval']['seed']+ti*10000 if mode=='greedy' else cfg['sample_eval']['seed']+ti*10000+ri*100

def validate_checkpoint(spec):
    p=checkpoint_path(spec);c=read(p/'complete.json')
    if file_hash(p/'complete.json')!=spec['complete_sha256'] or file_hash(p/'adapter/adapter_model.safetensors')!=spec['adapter_sha256']:raise ValueError('pinned diagnostic checkpoint differs')
    verify_files(p/'adapter',c['adapter_sha256'])
    if c['total_sft_updates']!=spec['total_sft_updates'] or c['total_dpo_updates']!=spec['total_dpo_updates']:raise ValueError('diagnostic ancestry differs')
    if spec['new_dpo_steps']:
        s=spec['new_dpo_steps']
        if c['algorithm']!='hard_recovery_weighted_dpo' or c['steps']!=s or c['cursor']!=169*(s//43) or c['epochs_completed']!=s//43 or c['source_sft_updates']!=255 or c['source_dpo_updates']!=11:raise ValueError('DPO milestone metadata differs')
    return {'complete.json':file_hash(p/'complete.json'),**{'adapter/'+n:h for n,h in c['adapter_sha256'].items()}}

def verify_inputs(output):
    cfg=read(output/'config.json');b=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs'/CONFIG) or cfg['training_enabled'] is not False or cfg['optimizer_updates']!=0:raise ValueError('fixed inference-only diagnosis required')
    if b['training_enabled'] is not False or b['optimizer_updates']!=0:raise ValueError('inference binding required')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256']);verify_files(ROOT,b['provenance'])
    for branch,spec in cfg['checkpoints'].items():verify_files(checkpoint_path(spec),b['checkpoints'][branch])
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',b['model_files'])
    for n,key in [('config.json','config_sha256'),('pairs.jsonl','pairs_sha256'),('reference_scores.json','reference_scores_sha256'),('weights.json','weights_sha256'),('probes.jsonl','probes_sha256'),('recovery_contexts.jsonl','contexts_sha256'),('development_tasks.json','tasks_sha256')]:
        if file_hash(output/n)!=b[key]:raise ValueError('diagnostic input differs: '+n)
    return cfg

def prepare(output):
    cfg=read(output/'config.json')
    if cfg!=read(ROOT/'configs'/CONFIG) or cfg['training_enabled'] is not False or cfg['optimizer_updates']!=0:raise ValueError('fixed inference-only config required')
    origin=ROOT/'artifacts'/cfg['origin_run'];oc=read(origin/'config.json');ob=read(origin/'input_binding.json')
    for name,key in [('config.json','origin_config_sha256'),('input_binding.json','origin_binding_sha256'),('reference_scores.json','reference_scores_sha256'),('pairs.jsonl','pairs_sha256')]:
        if file_hash(origin/name)!=cfg[key]:raise ValueError('diagnosis origin differs: '+name)
    if not read(origin/'audit.json')['passed'] or read(origin/'run_status.json')['error'] is not None:raise ValueError('audited completed origin required')
    verify_files(origin/'source',read(origin/'run_manifest.json')['source_sha256'])
    verify_matched_updates(oc,read(origin/'final/complete.json'),read(origin/'training_summary.json'))
    pairs=load(origin/'pairs.jsonl');validate_training_pairs(pairs);weights,ws=recovery_weights(pairs,oc['hard_weight'])
    if read(origin/'weights.json')!={'weights':weights,'summary':ws}:raise ValueError('original normalized weights differ')
    bound={};provenance=[origin/n for n in ('config.json','input_binding.json','audit.json','run_status.json','run_manifest.json','reference_scores.json','pairs.jsonl','weights.json','training_summary.json','before_pair_scores.json','after_pair_scores.json')]
    for branch,spec in cfg['checkpoints'].items():
        bound[branch]=validate_checkpoint(spec);p=checkpoint_path(spec)
        if spec['new_dpo_steps'] and read(p/'binding.json')!=ob:raise ValueError('milestone training binding differs')
        target=output/'checkpoints'/branch;target.mkdir(parents=True);shutil.copyfile(p/'complete.json',target/'complete.json')
        provenance.extend([p/'complete.json'])
        if not spec['new_dpo_steps']:
            prior=p.parent
            if not read(prior/'audit.json')['passed']:raise ValueError('audited comparison required')
            provenance.extend([prior/'audit.json',prior/'model_identity.json'])
    identity=read(origin/'model_identity.json')
    if identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('fixed base revision required')
    for spec in cfg['checkpoints'].values():
        if not spec['new_dpo_steps'] and read(checkpoint_path(spec).parent/'model_identity.json')!=identity:raise ValueError('comparison base identity differs')
    folder=ROOT/'.cache/qwen2.5-1.5b-instruct';verify_files(folder,identity['files_sha256'])
    for name in ('pairs.jsonl','weights.json','reference_scores.json'):shutil.copyfile(origin/name,output/name)
    write(output/'model_identity.json',identity);cs=investigation_contexts();probes=build_probes();jsonl(output/'recovery_contexts.jsonl',cs);jsonl(output/'probes.jsonl',probes);write(output/'development_tasks.json',[asdict(t) for t in investigation_tasks()])
    for n in identity['files_sha256']:
        if n.endswith(('.json','.txt','.jinja')):
            target=output/'tokenizer'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(folder/n,target)
    provenance.append(origin/'model_identity.json')
    for p in set(provenance):
        target=output/'provenance'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write(output/'input_binding.json',{'training_enabled':False,'optimizer_updates':0,'checkpoints':bound,'model_files':identity['files_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in sorted(set(provenance))},'config_sha256':file_hash(output/'config.json'),
        'pairs_sha256':file_hash(output/'pairs.jsonl'),'reference_scores_sha256':file_hash(output/'reference_scores.json'),'weights_sha256':file_hash(output/'weights.json'),
        'contexts_sha256':file_hash(output/'recovery_contexts.jsonl'),'probes_sha256':file_hash(output/'probes.jsonl'),'tasks_sha256':file_hash(output/'development_tasks.json')})
    write(output/'prepare_summary.json',{'checkpoints':len(bound),'training_pairs':len(pairs),'teacher_forced_probes':len(probes),'greedy_recovery_contexts':len(cs),
        'complete_episodes':len(bound)*4*(1+cfg['sample_eval']['rollouts_per_task']),'recovery_episodes':len(bound)*len(cs),'training_enabled':False,'optimizer_updates':0,'test_split_used':False})
    verify_inputs(output)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);prepare(p.parse_args().output)
