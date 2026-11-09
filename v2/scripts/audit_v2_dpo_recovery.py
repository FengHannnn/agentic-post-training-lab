"""Audit recovery DPO inputs, reference, optimizer and held-out recovery/task evaluations."""
import sys,json,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from evaluate_v2_dpo_recovery import verify_inputs
from apt_v2.tasks import make_tasks
from apt_v2.dpo_recovery import SCENARIOS,build_context,encode_recovery_pair
from apt_v2.dpo_recovery_training import validate_training_pairs,verify_reference,recovery_result,summarize_recovery,verify_update_history
from apt_v2.dpo_test import replay_episode,paired_outcomes
from apt_v2.sft_diagnostics import summarize_episodes

def load(path):return [json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]

def paired(before,after,sampled=False):
    result=paired_outcomes(before,after,sampled)
    result['changes']=[{'task_id':c['task_id'],'rollout_index':c['rollout_index'],'before_success':c['sft_success'],'after_success':c['dpo_success']} for c in result['changes']]
    return result

def audit_evaluations(output,cfg,tok):
    tasks=make_tasks();tm={t.id:t for t in tasks};comparisons=read(output/'comparison_summary.json');sample_comparison=read(output/'stochastic_comparison.json')
    branches=tuple(cfg.get('evaluation_branches',('before','after')))
    all_episodes={};episode_count=turns=0
    for branch in branches:
        all_episodes[branch]={}
        for split,count in cfg['eval']['splits'].items():
            path=output/(branch+'-'+split+'-episodes.jsonl');episodes=load(path);expected={t.id for t in tasks if t.split==split}
            if len(episodes)!=count or {e['task_id'] for e in episodes}!=expected:raise ValueError('complete distinct greedy task set required')
            for e in episodes:
                if e['model_branch']!=branch:raise ValueError('greedy branch differs')
                turns+=replay_episode(tm[e['task_id']],e,tok,'greedy');episode_count+=1
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)}
            if summary!=comparisons[branch][split] or summary!=read(output/(branch+'-'+split+'-summary.json')):raise ValueError('greedy summary differs')
            all_episodes[branch][split]=episodes
        path=output/(branch+'-sample-episodes.jsonl');episodes=load(path);dev=[t for t in tasks if t.split=='development'];sc=cfg['sample_eval']
        expected={(t.id,i) for t in dev for i in range(sc['rollouts_per_task'])}
        if len(dev)!=sc['task_count'] or len(episodes)!=len(expected) or {(e['task_id'],e['rollout_index']) for e in episodes}!=expected:raise ValueError('complete distinct development sample set required')
        for e in episodes:
            seed=sc['seed']+[t.id for t in dev].index(e['task_id'])*10000+e['rollout_index']*100
            if e['model_branch']!=branch or e['split']!='development' or e['sample_seed']!=seed or e['temperature']!=sc['temperature']:raise ValueError('development sample identity differs')
            if any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):raise ValueError('episode sampling seed differs')
            turns+=replay_episode(tm[e['task_id']],e,tok,'dpo_sample');episode_count+=1
        summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),'successes_by_task':{t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in dev}}
        if summary!=sample_comparison[branch] or summary!=read(output/(branch+'-sample-summary.json')):raise ValueError('sample summary differs')
        all_episodes[branch]['sample']=episodes
    dev=[t for t in tasks if t.split=='development'];contexts=[build_context(t,s,allowed_split='development') for t in dev for s in SCENARIOS]
    if contexts!=load(output/'recovery_contexts.jsonl'):raise ValueError('development recovery context manifest differs')
    cm={c['context_id']:c for c in contexts};rc=cfg['recovery_eval'];recovery_comparison=read(output/'recovery_comparison.json');action_count=0
    expected={(c['context_id'],mode,i) for c in contexts for mode in ('greedy','sample') for i in range(1 if mode=='greedy' else rc['samples_per_context'])}
    for branch in branches:
        path=output/(branch+'-recovery-actions.jsonl');rows=load(path)
        if len(rows)!=len(expected) or {(r['context_id'],r['mode'],r['sample_index']) for r in rows}!=expected:raise ValueError('complete distinct recovery action set required')
        for row in rows:
            c=cm[row['context_id']];seed=rc['seed']+[t.id for t in dev].index(c['task_id'])*10000+list(SCENARIOS).index(c['scenario'])*100+row['sample_index']
            if row['model_branch']!=branch or row['split']!='development' or row['task_id']!=c['task_id'] or row['family_id']!=c['family_id'] or row['scenario']!=c['scenario'] or row['sample_seed']!=seed or row['temperature']!=(0 if row['mode']=='greedy' else rc['temperature']):raise ValueError('recovery action identity differs')
            actual=recovery_result(tm[c['task_id']],c,row['generation'],tok,'greedy' if row['mode']=='greedy' else 'dpo_sample')
            if actual!=row['result']:raise ValueError('recovery result differs')
            action_count+=1
        summary={mode:{'overall':summarize_recovery([r for r in rows if r['mode']==mode]),'by_scenario':{s:summarize_recovery([r for r in rows if r['mode']==mode and r['scenario']==s]) for s in SCENARIOS}} for mode in ('greedy','sample')}
        summary['actions_sha256']=file_hash(path)
        if summary!=recovery_comparison[branch]:raise ValueError('recovery comparison differs')
    evidence={'episodes_replayed':episode_count,'episode_turns_replayed':turns,'development_recovery_contexts':len(contexts),'recovery_actions_replayed':action_count,
        'paired_greedy':{s:paired(all_episodes['before'][s],all_episodes['after'][s]) for s in cfg['eval']['splits']},
        'paired_development_sample':paired(all_episodes['before']['sample'],all_episodes['after']['sample'],True)}

    if 'dpo' in branches:
        evidence['paired_dpo_vs_sft_greedy']={s:paired(all_episodes['dpo'][s],all_episodes['after'][s]) for s in cfg['eval']['splits']}
        evidence['paired_dpo_vs_sft_development_sample']=paired(all_episodes['dpo']['sample'],all_episodes['after']['sample'],True)
    return evidence

def audit(output):
    import torch
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);pairs=load(output/'pairs.jsonl')
    if file_hash(output/'pairs.jsonl')!=cfg['pairs_sha256']:raise ValueError('reviewed recovery pairs changed')
    validate_training_pairs(pairs)
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs]
    verify_reference(pairs,encoded,read(output/'reference_scores.json'),cfg['adapter_sha256'])
    for name in ('before_pair_scores.json','after_pair_scores.json'):
        scores=read(output/name);metrics=scores['metrics']
        if scores['pairs']!=169 or len(metrics)!=169:raise ValueError('pair metric count differs')
        for k,v in scores['means'].items():
            actual=sum(m[k] for m in metrics)/len(metrics)
            if not math.isfinite(actual) or abs(actual-v)>1e-8:raise ValueError('pair metric mean differs')
        for m in metrics:
            if not all(math.isfinite(v) for v in m.values()) or abs(m['loss']-math.log1p(math.exp(-cfg['beta']*m['relative_margin'])))>1e-5:raise ValueError('DPO pair objective differs')
    if abs(read(output/'before_pair_scores.json')['means']['loss']-math.log(2))>1e-5:raise ValueError('initial policy/reference mismatch')
    final=read(output/'final/complete.json');train=read(output/'training_summary.json')
    if file_hash(output/'final/state.pt')!=final['state_sha256']:raise ValueError('optimizer state differs')
    state=torch.load(output/'final/state.pt',map_location='cpu',weights_only=True)
    verify_update_history(cfg,final,train,state)
    if cfg['epochs']>1:
        if load(output/'training_steps.jsonl')!=train['history']:raise ValueError('streamed training metrics differ')
        for step,epoch in ((43,1),(86,2)):
            checkpoint=output/f'checkpoint-{step:04}';complete=read(checkpoint/'complete.json')
            if complete['steps']!=step or complete['cursor']!=169*epoch or complete['epochs_completed']!=epoch:raise ValueError('epoch checkpoint differs')
            verify_files(checkpoint/'adapter',complete['adapter_sha256'])
    evidence=audit_evaluations(output,cfg,tok)
    write(output/'evaluation_summary.json',{**evidence,'status':'awaiting_local_review','optimizer_updates':cfg['max_steps'],'recovery_comparison':read(output/'recovery_comparison.json')})
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,**{k:v for k,v in evidence.items() if k not in ('paired_greedy','paired_development_sample')},
        'preference_pairs':169,'reference_inputs_verified':169,'optimizer_updates':cfg['max_steps'],'reference_and_initial_checkpoint_unchanged':True,'test_split_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
