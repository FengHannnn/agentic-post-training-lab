"""Inference-only independent test using the established paired vLLM evaluator."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.sft_test import holdout_tasks,holdout_summary,acceptance
from apt_v2.sql_sft import audit_episode,data_identity

def checkpoints(cfg):
    return {'parent':ROOT/'artifacts'/cfg['parent_run']/'final',
        'weighted':ROOT/'artifacts'/cfg['candidate_run']/'final'}

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    for branch,path in checkpoints(cfg).items():verify_files(path,binding['checkpoints'][branch])
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files'])
    candidate_examples=ROOT/'artifacts'/cfg['candidate_run']/'examples.jsonl'
    if file_hash(candidate_examples)!=binding['candidate_examples_sha256']:raise ValueError('candidate training provenance changed')
    return cfg

def prepare(output):
    cfg=read(output/'config.json');candidate=ROOT/'artifacts'/cfg['candidate_run']
    parent=ROOT/'artifacts'/cfg['parent_run'];paths=checkpoints(cfg);bindings={}
    for branch,path in paths.items():
        complete=read(path/'complete.json');verify_files(path/'adapter',complete['adapter_sha256'])
        expected=cfg['parent_adapter_sha256'] if branch=='parent' else cfg['candidate_adapter_sha256']
        if file_hash(path/'adapter/adapter_model.safetensors')!=expected:raise ValueError('pinned adapter differs: '+branch)
        bindings[branch]={**{'adapter/'+n:h for n,h in complete['adapter_sha256'].items()},
            'complete.json':file_hash(path/'complete.json')}
    if read(paths['weighted']/'complete.json')['total_steps']!=126:raise ValueError('126-step candidate required')
    identity=read(candidate/'model_identity.json');folder=ROOT/'.cache/qwen2.5-1.5b-instruct'
    if identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('fixed base model required')
    verify_files(folder,identity['files_sha256'])
    examples_file=candidate/'examples.jsonl';examples=[json.loads(l) for l in examples_file.read_text(encoding='utf8').splitlines()]
    if examples_file.read_bytes()!=(parent/'examples.jsonl').read_bytes():raise ValueError('unchanged training data required')
    if data_identity(examples)!=read(candidate/'input_binding.json')['dataset_sha256']:
        raise ValueError('candidate dataset identity differs')
    tasks=holdout_tasks(cfg,examples)
    from dataclasses import asdict
    write(output/'test_tasks.json',[asdict(t) for t in tasks])
    write(output/'model_identity.json',identity)
    write(output/'input_binding.json',{'checkpoints':bindings,'model_files':identity['files_sha256'],
        'candidate_examples_sha256':file_hash(examples_file),'candidate_run':cfg['candidate_run'],
        'training_enabled':False,'optimizer_updates':0})
    # Existing evaluator expects the candidate under output/final. Copy only adapter evidence.
    (output/'final').mkdir()
    shutil.copytree(paths['weighted']/'adapter',output/'final/adapter')
    shutil.copyfile(paths['weighted']/'complete.json',output/'final/complete.json')
    write(output/'prepare_summary.json',{'tasks':len(tasks),'fault_tasks':sum(t.fault_mode!='none' for t in tasks),
        'parent_adapter_sha256':cfg['parent_adapter_sha256'],'candidate_adapter_sha256':cfg['candidate_adapter_sha256'],
        'candidate_total_steps':126,'training_enabled':False,'optimizer_updates':0})

def evaluate(output):
    cfg=verify_inputs(output)
    from evaluate_v2_sql_sft import evaluate as paired_evaluate
    paired_evaluate(output)
    verify_inputs(output)

def audit(output):
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);tasks=read(output/'test_tasks.json')
    from apt_v2.tasks import make_tasks
    task_map={t.id:t for t in make_tasks() if t.split=='test'}
    if set(task_map)!={t['id'] for t in tasks}:raise ValueError('fixed test manifest differs')
    from dataclasses import asdict
    if [asdict(t) for t in task_map.values()]!=tasks:raise ValueError('test task content differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    for name,h in read(output/'model_identity.json')['files_sha256'].items():
        p=output/'tokenizer'/name
        if p.exists() and file_hash(p)!=h:raise ValueError('saved tokenizer differs')
    summaries={};turns=0
    comparison=read(output/'comparison_summary.json')
    for branch in ('parent','weighted'):
        path=output/(branch+'-test-episodes.jsonl')
        episodes=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
        for e in episodes:
            if e['model_branch']!=branch:raise ValueError('model branch differs')
            turns+=audit_episode(task_map[e['task_id']],e,tok)
        summary=holdout_summary(episodes,list(task_map))
        original={k:v for k,v in summary.items() if k not in ('valid_checks','valid_submits')}
        original['episodes_sha256']=file_hash(path)
        if original!=read(output/(branch+'-test-summary.json')) or original!=comparison[branch]['test']:
            raise ValueError('test summary differs')
        summaries[branch]=summary
    decision=acceptance(summaries['weighted'],cfg['acceptance'])
    write(output/'independent_test_summary.json',{'models':summaries,'acceptance':decision,
        'candidate_adapter_sha256':cfg['candidate_adapter_sha256'],'training_enabled':False,'optimizer_updates':0,
        'promotion_status':'awaiting_local_review'})
    write(output/'audit.json',{'passed':True,'episodes_replayed':8,'turns_replayed':turns,
        'checkpoints_unchanged':True,'optimizer_updates':0})
    print(json.dumps(decision),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate','audit'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    globals()[a.stage](a.output)
