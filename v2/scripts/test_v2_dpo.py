"""Inference-only independent SFT/DPO test: pinned inputs, paired evaluation and replay."""
import sys,json,shutil
from pathlib import Path
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.tasks import make_tasks
from apt_v2.sft_test import holdout_tasks
from apt_v2.sql_sft import data_identity
from apt_v2.dpo_test import replay_episode,validate_set,paired_outcomes

def checkpoints(cfg):
    return {'sft':ROOT/'artifacts'/cfg['parent_run']/'final','dpo':ROOT/'artifacts'/cfg['candidate_run']/'final'}

def verify_inputs(output):
    cfg=read(output/'config.json');binding=read(output/'input_binding.json')
    if cfg!=read(ROOT/'configs/dpo-independent-test-v1.json'):raise ValueError('fixed evaluation config differs')
    verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    for branch,path in checkpoints(cfg).items():verify_files(path,binding['checkpoints'][branch])
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',binding['model_files'])
    for rel,h in binding['provenance'].items():
        if file_hash(ROOT/rel)!=h:raise ValueError('training provenance changed: '+rel)
    verify_files(output/'final',binding['checkpoints']['dpo'])
    tasks=[t for t in make_tasks() if t.split=='test']
    if read(output/'test_tasks.json')!=[asdict(t) for t in tasks]:raise ValueError('test task manifest differs')
    return cfg

def prepare(output):
    cfg=read(output/'config.json');parent=ROOT/'artifacts'/cfg['parent_run'];candidate=ROOT/'artifacts'/cfg['candidate_run']
    if cfg['training_enabled'] is not False or cfg['sample_eval']['task_count']!=4:raise ValueError('inference-only fixed test required')
    bindings={}
    for branch,path in checkpoints(cfg).items():
        complete=read(path/'complete.json');verify_files(path/'adapter',complete['adapter_sha256'])
        expected=cfg['adapter_sha256'] if branch=='sft' else cfg['candidate_adapter_sha256']
        if file_hash(path/'adapter/adapter_model.safetensors')!=expected:raise ValueError('pinned adapter differs: '+branch)
        bindings[branch]={**{'adapter/'+n:h for n,h in complete['adapter_sha256'].items()},'complete.json':file_hash(path/'complete.json')}
    if read(parent/'final/complete.json')['total_steps']!=126:raise ValueError('126-step SFT required')
    final=read(candidate/'final/complete.json');training=read(candidate/'training_summary.json')
    if final['algorithm']!='dpo' or final['steps']!=11 or training['optimizer_updates']!=11:raise ValueError('reviewed 11-step DPO required')
    dpo_cfg=read(candidate/'config.json')
    if dpo_cfg['adapter_sha256']!=cfg['adapter_sha256'] or dpo_cfg['pairs_sha256']!=cfg['pairs_sha256']:raise ValueError('DPO provenance differs')
    identity=read(parent/'model_identity.json')
    if identity!=read(candidate/'model_identity.json') or identity['resolved_revision']!='989aa7980e4cf806f80c7fef2b1adb7bc71aa306':raise ValueError('fixed base model differs')
    verify_files(ROOT/'.cache/qwen2.5-1.5b-instruct',identity['files_sha256'])
    examples_file=parent/'examples.jsonl';examples=[json.loads(l) for l in examples_file.read_text(encoding='utf8').splitlines()]
    if data_identity(examples)!=read(parent/'input_binding.json')['dataset_sha256']:raise ValueError('SFT dataset differs')
    tasks=holdout_tasks(cfg,examples);allowed={(t.id,t.family_id) for t in make_tasks() if t.split=='train'}
    pairs_file=candidate/'pairs.jsonl';original=ROOT/'artifacts'/cfg['data_run']/'pairs.jsonl'
    if file_hash(pairs_file)!=cfg['pairs_sha256'] or file_hash(original)!=cfg['pairs_sha256']:raise ValueError('reviewed pairs differ')
    pairs=[json.loads(l) for l in pairs_file.read_text(encoding='utf8').splitlines()]
    if len(pairs)!=44 or any(p['split']!='train' or (p['task_id'],p['family_id']) not in allowed for p in pairs):raise ValueError('train-only DPO provenance required')
    provenance=[examples_file,parent/'input_binding.json',parent/'model_identity.json',candidate/'model_identity.json',candidate/'config.json',candidate/'training_summary.json',pairs_file,original]
    write(output/'input_binding.json',{'checkpoints':bindings,'model_files':identity['files_sha256'],
        'provenance':{p.relative_to(ROOT).as_posix():file_hash(p) for p in provenance},'training_enabled':False,'optimizer_updates':0})
    write(output/'test_tasks.json',[asdict(t) for t in tasks]);write(output/'model_identity.json',identity)
    (output/'final').mkdir();shutil.copytree(candidate/'final/adapter',output/'final/adapter')
    shutil.copyfile(candidate/'final/complete.json',output/'final/complete.json')
    for name in identity['files_sha256']:
        if name.endswith(('.json','.txt','.jinja')):
            p=output/'tokenizer'/name;p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/'.cache/qwen2.5-1.5b-instruct'/name,p)
    write(output/'prepare_summary.json',{'test_tasks':4,'fault_tasks':2,'greedy_episodes':8,
        'sample_episodes':2*4*cfg['sample_eval']['rollouts_per_task'],'training_enabled':False,'optimizer_updates':0})
    verify_inputs(output)

def greedy(output):
    verify_inputs(output)
    from evaluate_v2_dpo import evaluate
    evaluate(output);verify_inputs(output)

def sample(output):
    verify_inputs(output)
    from evaluate_v2_dpo import sample as evaluate_sample
    evaluate_sample(output,split='test');verify_inputs(output)

def audit(output):
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);tasks=[t for t in make_tasks() if t.split=='test'];task_map={t.id:t for t in tasks}
    for name,h in read(output/'model_identity.json')['files_sha256'].items():
        if (output/'tokenizer'/name).exists() and file_hash(output/'tokenizer'/name)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    all_episodes={};turns=0;count=0
    comparisons=read(output/'comparison_summary.json');stochastic=read(output/'stochastic_comparison.json')
    for branch in ('sft','dpo'):
        all_episodes[branch]={}
        for mode,filename in [('greedy',branch+'-test-episodes.jsonl'),('sample',branch+'-sample-episodes.jsonl')]:
            path=output/filename;episodes=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
            summary=validate_set(episodes,tasks,branch,cfg['sample_eval'] if mode=='sample' else None)
            for e in episodes:turns+=replay_episode(task_map[e['task_id']],e,tok,'greedy' if mode=='greedy' else 'dpo_sample');count+=1
            summary['episodes_sha256']=file_hash(path)
            if mode=='sample':summary['successes_by_task']={t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in tasks}
            expected=comparisons[branch]['test'] if mode=='greedy' else stochastic[branch]
            saved=read(output/(branch+'-test-summary.json' if mode=='greedy' else branch+'-sample-summary.json'))
            if summary!=expected or summary!=saved:raise ValueError('test summary differs')
            all_episodes[branch][mode]=episodes
    write(output/'independent_test_summary.json',{'greedy':comparisons,'sample':stochastic,
        'paired_greedy':paired_outcomes(all_episodes['sft']['greedy'],all_episodes['dpo']['greedy']),
        'paired_sample':paired_outcomes(all_episodes['sft']['sample'],all_episodes['dpo']['sample'],True),
        'unique_test_tasks':4,'sample_rollouts_per_task':cfg['sample_eval']['rollouts_per_task'],
        'training_enabled':False,'optimizer_updates':0,'status':'awaiting_local_review'})
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,'episodes_replayed':count,'turns_replayed':turns,'checkpoints_unchanged':True,
        'train_test_task_family_disjoint':True,'training_enabled':False,'optimizer_updates':0})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','greedy','sample','audit'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.stage](a.output)
