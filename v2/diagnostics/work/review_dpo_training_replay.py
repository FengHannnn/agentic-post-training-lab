"""Replay greedy and stochastic SFT/DPO evaluations; verify fixed reference inputs."""
import sys,json,math
from pathlib import Path
sys.path[:0]=[r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/src',r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/scripts']
from common import ROOT,read,write,file_hash,verify_files
from apt_v2.sql_sft import audit_episode
from apt_v2.dpo import verified_pairs,encode_pair
from apt_v2.dpo_training import input_id
from apt_v2.tasks import make_tasks
from apt_v2.sft_diagnostics import summarize_episodes

def audit(output):
    from transformers import AutoTokenizer
    cfg=read(output/'config.json');verify_files(ROOT,read(output/'run_manifest.json')['source_sha256'])
    parent=ROOT/'artifacts/reviews/sql-sft-weighted-v1-20261101/extracted/sql-sft-weighted-v1-20261101'
    if file_hash(parent/'final/adapter/adapter_model.safetensors')!=cfg['adapter_sha256']:raise ValueError('SFT baseline changed')
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    task_map={t.id:t for t in make_tasks()};turns=episodes_count=0
    comparisons=read(output/'comparison_summary.json');sample_comparisons=read(output/'stochastic_comparison.json')
    for branch in ('sft','dpo'):
        for split,count in cfg['eval']['splits'].items():
            path=output/(branch+'-'+split+'-episodes.jsonl')
            episodes=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
            if len(episodes)!=count or {e['task_id'] for e in episodes}!={t.id for t in task_map.values() if t.split==split}:raise ValueError('complete distinct greedy task set required')
            for e in episodes:
                if e['model_branch']!=branch:raise ValueError('greedy branch differs')
                turns+=audit_episode(task_map[e['task_id']],e,tok);episodes_count+=1
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)}
            if summary!=comparisons[branch][split] or summary!=read(output/(branch+'-'+split+'-summary.json')):raise ValueError('greedy summary differs')
        path=output/(branch+'-sample-episodes.jsonl')
        episodes=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
        tasks=[t for t in make_tasks() if t.split=='train'];sc=cfg['sample_eval']
        expected={(t.id,i) for t in tasks for i in range(sc['rollouts_per_task'])}
        if len(episodes)!=len(expected) or {(e['task_id'],e['rollout_index']) for e in episodes}!=expected:raise ValueError('complete distinct sample set required')
        for e in episodes:
            ti=[t.id for t in tasks].index(e['task_id']);seed=sc['seed']+ti*10000+e['rollout_index']*100
            if e['model_branch']!=branch or e['sample_seed']!=seed or e['temperature']!=sc['temperature']:raise ValueError('sample identity differs')
            if any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):raise ValueError('sample seed differs')
            if verified_pairs(task_map[e['task_id']],e,tok,cfg['max_length'],0):raise ValueError('audit must not create training preferences')
            turns+=len(e['turns']);episodes_count+=1
        summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),
            'successes_by_task':{t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in tasks}}
        if summary!=sample_comparisons[branch] or summary!=read(output/(branch+'-sample-summary.json')):raise ValueError('sample summary differs')
    pairs=[json.loads(l) for l in (output/'pairs.jsonl').read_text(encoding='utf8').splitlines()]
    if file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed pairs changed')
    reference=read(output/'reference_scores.json')
    if reference['baseline_adapter_sha256']!=cfg['adapter_sha256'] or reference['pair_ids']!=[p['pair_id'] for p in pairs]:raise ValueError('reference policy/pair binding differs')
    if len(reference['scores'])!=len(pairs):raise ValueError('reference score count differs')
    for pair,ref in zip(pairs,reference['scores']):
        enc=encode_pair(tok,pair,cfg['max_length'])
        if ref['input_id']!=input_id(enc) or len(ref['chosen'])!=len(enc['chosen']) or len(ref['rejected'])!=len(enc['rejected']):raise ValueError('reference token binding differs')
        if any(not math.isfinite(v) for v in ref['chosen']+ref['rejected']):raise ValueError('finite reference required')
    for name in ('before_pair_scores.json','after_pair_scores.json'):
        score=read(output/name);metrics=score['metrics']
        if len(metrics)!=len(pairs):raise ValueError('pair score count differs')
        for key,value in score['means'].items():
            actual=sum(m[key] for m in metrics)/len(metrics)
            if not math.isfinite(actual) or abs(actual-value)>1e-8:raise ValueError('pair metric mean differs')
    if abs(read(output/'before_pair_scores.json')['means']['loss']-math.log(2))>1e-5:raise ValueError('initial DPO/reference mismatch')
    final=read(output/'final/complete.json');verify_files(output/'final/adapter',final['adapter_sha256'])
    assert not (output/'final/state.pt').exists() # Optimizer state retained on cloud and excluded from review ZIP.
    train=read(output/'training_summary.json')
    if train['optimizer_updates']!=11 or train['optimizer_start']!='fresh_adam' or final['steps']!=11:raise ValueError('DPO update count differs')
    write(Path(r'C:/Users/app/Documents/Codex/2026-9-28/AgenticPostTrainingLab/v2/artifacts/reviews/dpo-train-v1-20261102/local_audit.json'),{'passed':True,'episodes_replayed':episodes_count,'turns_replayed':turns,
        'preference_pairs':len(pairs),'reference_inputs_verified':len(pairs),'baseline_unchanged':True,'audit_optimizer_updates':0})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
