"""Audit scenario-weighted SFT updates, task rollouts and per-epoch recovery probes."""
import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash,verify_files
from evaluate_v2_hard_recovery_sft import verify_inputs,load
from diagnose_v2_recovery_contexts import audit_diagnostics,export_csv,sampling_seed,selected_tasks
from apt_v2.tasks import make_tasks
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.hard_recovery_sft import recovery_weights,verify_weighted_updates
from apt_v2.dpo_recovery import SCENARIOS
from apt_v2.recovery_diagnostics import diagnose_action,summarize,comparison_rates
from apt_v2.dpo_recovery_training import validate_generation
from apt_v2.dpo_test import replay_episode
from apt_v2.sft_diagnostics import summarize_episodes
from audit_v2_dpo_recovery import paired

def audit_task_evaluations(output,cfg,tok):
    tasks=make_tasks();tm={t.id:t for t in tasks};comparison=read(output/'comparison_summary.json');samples=read(output/'stochastic_comparison.json')
    all_episodes={};count=turns=0
    for branch in cfg['evaluation_branches']:
        all_episodes[branch]={}
        for split,n in cfg['eval']['splits'].items():
            path=output/(branch+'-'+split+'-episodes.jsonl');episodes=load(path);expected={t.id for t in tasks if t.split==split}
            if len(episodes)!=n or {e['task_id'] for e in episodes}!=expected:raise ValueError('complete distinct task evaluations required')
            for e in episodes:
                if e['model_branch']!=branch:raise ValueError('task branch differs')
                turns+=replay_episode(tm[e['task_id']],e,tok,'greedy');count+=1
            summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path)}
            if summary!=comparison[branch][split] or summary!=read(output/(branch+'-'+split+'-summary.json')):raise ValueError('greedy task summary differs')
            all_episodes[branch][split]=episodes
        path=output/(branch+'-sample-episodes.jsonl');episodes=load(path);dev=[t for t in tasks if t.split=='development'];sc=cfg['sample_eval']
        expected={(t.id,i) for t in dev for i in range(sc['rollouts_per_task'])}
        if len(dev)!=sc['task_count'] or len(episodes)!=len(expected) or {(e['task_id'],e['rollout_index']) for e in episodes}!=expected:raise ValueError('complete distinct development samples required')
        for e in episodes:
            seed=sc['seed']+[t.id for t in dev].index(e['task_id'])*10000+e['rollout_index']*100
            if e['model_branch']!=branch or e['split']!='development' or e['sample_seed']!=seed or e['temperature']!=sc['temperature'] or any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):
                raise ValueError('development sample identity differs')
            turns+=replay_episode(tm[e['task_id']],e,tok,'dpo_sample');count+=1
        summary={**summarize_episodes(episodes),'episodes_sha256':file_hash(path),'successes_by_task':{t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in dev}}
        if summary!=samples[branch] or summary!=read(output/(branch+'-sample-summary.json')):raise ValueError('sample task summary differs')
        all_episodes[branch]['sample']=episodes
    return {'episodes_replayed':count,'episode_turns_replayed':turns,
        'paired_greedy':{s:paired(all_episodes['before'][s],all_episodes['after'][s]) for s in cfg['eval']['splits']},
        'paired_development_sample':paired(all_episodes['before']['sample'],all_episodes['after']['sample'],True)}

def audit_checkpoint_actions(output,cfg,tok):
    tasks=selected_tasks(cfg);tm={t.id:t for t in tasks};contexts=[c for c in load(output/'contexts.jsonl') if c['split']=='train'];cm={c['context_id']:c for c in contexts}
    expected=set(cm);summary=read(output/'checkpoint_diagnostics.json');count=0
    if set(summary)!={f'checkpoint-{s:04}' for s in cfg['checkpoint_steps']}:raise ValueError('checkpoint diagnostic branches differ')
    for step in cfg['checkpoint_steps']:
        branch=f'checkpoint-{step:04}';path=output/(branch+'-actions.jsonl');rows=load(path)
        if len(rows)!=len(expected) or {r['context_id'] for r in rows}!=expected:raise ValueError('complete distinct checkpoint contexts required')
        for row in rows:
            c=cm[row['context_id']]
            if row['model_branch']!=branch or row['task_id']!=c['task_id'] or row['family_id']!=c['family_id'] or row['split']!='train' or row['scenario']!=c['scenario'] or row['mode']!='greedy' or row['sample_index']!=0 or row['temperature']!=0 or row['sample_seed']!=sampling_seed(cfg,tasks,c,0):
                raise ValueError('checkpoint recovery identity differs')
            g=row['generation'];validate_generation(tok,c['messages'],g,'greedy')
            if len(g['response_token_ids'])>cfg['eval']['max_new_tokens'] or (g['finish_reason']=='stop' and g['response_token_ids'][-1]!=tok.eos_token_id):raise ValueError('checkpoint token budget or EOS differs')
            actual,d=diagnose_action(tm[c['task_id']],c,g['text'])
            if actual!=row['row'] or d!=row['diagnostic']:raise ValueError('checkpoint recovery replay differs')
        actual={'overall':summarize(rows),'by_scenario':{s:summarize([r for r in rows if r['scenario']==s]) for s in SCENARIOS},'actions_sha256':file_hash(path)}
        if actual!=summary[branch]:raise ValueError('checkpoint recovery summary differs')
        count+=len(rows)
    return count

def checkpoint_progress(output,cfg):
    summary=read(output/'diagnostic_summary.json');middle=read(output/'checkpoint_diagnostics.json');rows=[]
    for name,step in [('before',0),*[(f'checkpoint-{s:04}',s) for s in cfg['checkpoint_steps']],('after',cfg['max_steps'])]:
        values=summary[name]['train']['greedy'] if name in ('before','after') else middle[name]
        p=ROOT/'artifacts'/cfg['initial_run']/'final' if name=='before' else output/('final' if name=='after' else name)
        rows.append({'checkpoint':name,'new_sft_updates':step,'total_sft_updates':cfg['source_sft_updates']+step,
            'adapter_sha256':read(p/'complete.json')['adapter_sha256']['adapter_model.safetensors'],
            'train_greedy':{k:v for k,v in values.items() if k!='actions_sha256'}})
    return {'selection':'fixed_final','checkpoints':rows}

def audit(output):
    import torch
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);binding=read(output/'input_binding.json');pairs=load(output/'pairs.jsonl');examples=chosen_examples(pairs)
    if examples!=load(output/'examples.jsonl') or data_sha256(examples)!=binding['examples_sha256']:raise ValueError('chosen example identity differs')
    weights,ws=recovery_weights(pairs,cfg['hard_weight'])
    if read(output/'weights.json')!={'weights':weights,'summary':ws}:raise ValueError('scenario weights differ')
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False);encoded=encode_chosen(tok,examples,cfg['max_length'])
    if data_sha256(encoded)!=binding['encoded_sha256']:raise ValueError('chosen encoding differs')
    for s in encoded:
        n=s['prompt_length']
        if s['labels'][:n]!=[-100]*n or s['labels'][n:]!=s['input_ids'][n:]:raise ValueError('assistant-only supervision required')
    for name in ('before_chosen_scores.json','after_chosen_scores.json'):
        scores=read(output/name)
        if scores['examples']!=len(encoded) or len(scores['metrics'])!=len(encoded):raise ValueError('complete chosen scores required')
        for s,m in zip(encoded,scores['metrics']):
            probs=m['token_log_probs']
            if not math.isfinite(m['loss']) or m['input_id']!=data_sha256(s) or len(probs)!=s['response_length'] or any(not math.isfinite(v) or v>1e-6 for v in probs) or abs(-sum(probs)/len(probs)-m['loss'])>1e-6:raise ValueError('chosen score binding differs')
        if not math.isfinite(scores['mean_loss']) or not math.isfinite(scores['mean_weighted_loss']) or abs(sum(m['loss'] for m in scores['metrics'])/len(encoded)-scores['mean_loss'])>1e-8 or abs(sum(m['loss']*w for m,w in zip(scores['metrics'],weights))/len(encoded)-scores['mean_weighted_loss'])>1e-8:raise ValueError('weighted chosen mean differs')
    parent=ROOT/'artifacts'/cfg['initial_run'];parent_scores=read(parent/'after_chosen_scores.json');initial_scores=read(output/'before_chosen_scores.json')
    if parent_scores['examples']!=len(encoded) or len(parent_scores['metrics'])!=len(encoded):raise ValueError('parent chosen scores differ')
    for a,b in zip(initial_scores['metrics'],parent_scores['metrics']):
        if a['input_id']!=b['input_id'] or len(a['token_log_probs'])!=len(b['token_log_probs']) or any(abs(x-y)>1e-5 for x,y in zip(a['token_log_probs'],b['token_log_probs'])):raise ValueError('weighted continuation initial probabilities differ')
    final=read(output/'final/complete.json');train=read(output/'training_summary.json')
    if file_hash(output/'final/state.pt')!=final['state_sha256']:raise ValueError('weighted optimizer state differs')
    state=torch.load(output/'final/state.pt',map_location='cpu',weights_only=True);verify_weighted_updates(cfg,final,train,state)
    if load(output/'training_steps.jsonl')!=train['history'] or train['weight_summary']!=ws or train['rejected_used_in_loss'] is not False or train['reference_used_in_loss'] is not False or train['token_weighting'] is not False or train['supervised_tokens_per_epoch']!=sum(s['response_length'] for s in encoded):raise ValueError('weighted supervision evidence differs')
    for step in cfg['checkpoint_steps']:
        p=output/f'checkpoint-{step:04}';c=read(p/'complete.json')
        if c['algorithm']!='hard_recovery_weighted_sft' or c['steps']!=step or c['epochs_completed']!=step//cfg['checkpoint_every'] or c['cursor']!=cfg['pair_count']*(step//cfg['checkpoint_every']) or c['total_sft_updates']!=cfg['source_sft_updates']+step or c['total_dpo_updates']!=cfg['source_dpo_updates']:raise ValueError('epoch checkpoint schedule differs')
        if file_hash(p/'state.pt')!=c['state_sha256']:raise ValueError('epoch optimizer state differs')
    tasks=audit_task_evaluations(output,cfg,tok)
    recovery=audit_diagnostics(output,{**cfg,'checkpoints':{b:{} for b in cfg['evaluation_branches']}},tok)
    checkpoint_actions=audit_checkpoint_actions(output,cfg,tok)
    progress=checkpoint_progress(output,cfg);write(output/'checkpoint_progress.json',progress)
    export_csv(output);write(output/'train_development_comparison.json',comparison_rates(read(output/'diagnostic_summary.json')))
    evidence={**tasks,'final_recovery_actions_replayed':recovery['actions_replayed'],'checkpoint_recovery_actions_replayed':checkpoint_actions,
        'train_contexts':recovery['train_contexts'],'development_contexts':recovery['development_contexts'],
        'training_contexts_exactly_in_pairs':recovery['training_contexts_exactly_in_pairs'],'development_contexts_in_training_pairs':0,
        'chosen_examples':len(encoded),'optimizer_updates':cfg['max_steps'],'hard_examples':ws['hard_examples'],'checkpoint_selection':'fixed_final'}
    write(output/'evaluation_summary.json',{**evidence,'status':'awaiting_local_review'})
    verify_inputs(output)
    write(output/'audit.json',{'passed':True,**{k:v for k,v in evidence.items() if not k.startswith('paired_')},
        'weights_verified':True,'chosen_only_supervision':True,'parent_initial_probabilities_match':True,'parent_checkpoint_unchanged':True,'test_split_used':False})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
