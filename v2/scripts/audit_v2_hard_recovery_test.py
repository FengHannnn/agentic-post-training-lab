"""Replay independent task/recovery results and add cross-month SQL challenges."""
import sys,json,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT,read,write,file_hash
from test_v2_hard_recovery import verify_inputs,load,recovery_seed,recovery_summary
from apt_v2.independent_recovery_test import independent_tasks,test_contexts,diagnose_test_action,replay_test_recovery,TEST_SCENARIOS,NOVEL_SCENARIOS,calendar_episode_audit,calendar_sql_check
from apt_v2.recovery_diagnostics import messages_id
from apt_v2.dpo_recovery_training import validate_generation
from apt_v2.dpo_test import replay_episode,validate_set
from audit_v2_dpo_recovery import paired

def check_native_budget(g,cfg,tok):
    ids=g['response_token_ids']
    if len(ids)>cfg['eval']['max_new_tokens'] or (g['finish_reason']=='stop' and ids[-1]!=tok.eos_token_id):raise ValueError('test native token budget or EOS differs')

def audit_results(output,cfg,tok):
    pairs=load(output/'pairs.jsonl');examples=load(output/'examples.jsonl');tasks=independent_tasks(pairs,examples);tm={t.id:t for t in tasks}
    from dataclasses import asdict
    if read(output/'test_tasks.json')!=[asdict(t) for t in tasks]:raise ValueError('test task manifest differs')
    contexts=test_contexts(tasks)
    if contexts!=load(output/'contexts.jsonl'):raise ValueError('independent recovery contexts differ')
    seen={messages_id(p['messages']) for p in pairs}
    if any(messages_id(c['messages']) in seen for c in contexts):raise ValueError('test messages overlap training')
    cm={c['context_id']:c for c in contexts};comparisons=read(output/'comparison_summary.json');stochastic=read(output/'stochastic_comparison.json')
    complete={};episode_count=turns=0;calendar_cases=[]
    for branch in cfg['evaluation_branches']:
        complete[branch]={}
        for mode,name in [('greedy',branch+'-test'),('sample',branch+'-sample')]:
            path=output/(name+'-episodes.jsonl');episodes=load(path)
            summary=validate_set(episodes,tasks,branch,cfg['sample_eval'] if mode=='sample' else None)
            for e in episodes:
                turns+=replay_episode(tm[e['task_id']],e,tok,'greedy' if mode=='greedy' else 'dpo_sample');episode_count+=1
                for row in e['turns']:check_native_budget(row['generation'],cfg,tok)
                calendar_cases.append({'kind':'complete','model_branch':branch,'mode':mode,'task_id':e['task_id'],'rollout_index':e.get('rollout_index',0),
                    'result':calendar_episode_audit(tm[e['task_id']],e['turns'],e['success'],e['artifact'])})
            summary['episodes_sha256']=file_hash(path)
            if mode=='sample':summary['successes_by_task']={t.id:sum(e['success'] for e in episodes if e['task_id']==t.id) for t in tasks}
            if summary!=(comparisons[branch]['test'] if mode=='greedy' else stochastic[branch]) or summary!=read(output/(name+'-summary.json')):
                raise ValueError('independent complete summary differs')
            complete[branch][mode]=episodes
    rc=cfg['recovery_eval'];expected={(c['context_id'],mode,i) for c in contexts for mode in ('greedy','sample') for i in range(1 if mode=='greedy' else rc['samples_per_context'])}
    expected_episodes={key for key in expected if cm[key[0]]['scenario'] in NOVEL_SCENARIOS}
    actions=[];continuations=[];continuation_turns=0;calendar_first=[]
    for branch in cfg['evaluation_branches']:
        rows=load(output/(branch+'-recovery-actions.jsonl'));episodes=load(output/(branch+'-recovery-episodes.jsonl'))
        if len(rows)!=len(expected) or {(r['context_id'],r['mode'],r['sample_index']) for r in rows}!=expected:raise ValueError('complete distinct test recovery actions required')
        if len(episodes)!=len(expected_episodes) or {(e['context_id'],e['mode'],e['sample_index']) for e in episodes}!=expected_episodes:raise ValueError('complete distinct test continuations required')
        rm={(r['context_id'],r['mode'],r['sample_index']):r for r in rows}
        for row in rows:
            c=cm[row['context_id']];mode=row['mode'];seed=recovery_seed(cfg,tasks,c,row['sample_index'])
            if row['model_branch']!=branch or row['task_id']!=c['task_id'] or row['family_id']!=c['family_id'] or row['split']!='test' or row['scenario']!=c['scenario'] or row['sample_seed']!=seed or row['temperature']!=(0 if mode=='greedy' else rc['temperature']):
                raise ValueError('independent recovery identity differs')
            g=row['generation'];validate_generation(tok,c['messages'],g,'greedy' if mode=='greedy' else 'dpo_sample');check_native_budget(g,cfg,tok)
            actual,d=diagnose_test_action(tm[c['task_id']],c,g['text'])
            if actual!=row['row'] or d!=row['diagnostic']:raise ValueError('independent recovery replay differs')
            if actual['action'].get('tool')=='query_sql' and actual['observation'].get('verified_metrics'):
                calendar_first.append({'model_branch':branch,'mode':mode,'context_id':c['context_id'],'sample_index':row['sample_index'],
                    'check':calendar_sql_check(tm[c['task_id']],actual['action']['arguments']['sql'],actual['observation']['verified_metrics'])})
        for e in episodes:
            c=cm[e['context_id']];mode=e['mode'];seed=recovery_seed(cfg,tasks,c,e['sample_index'])
            if e['model_branch']!=branch or e['sample_seed']!=seed or e['temperature']!=(0 if mode=='greedy' else rc['temperature']) or any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):
                raise ValueError('independent continuation seed or branch differs')
            continuation_turns+=replay_test_recovery(tm[c['task_id']],c,e,tok,'greedy' if mode=='greedy' else 'dpo_sample')
            for row in e['turns']:check_native_budget(row['generation'],cfg,tok)
            first=rm[(e['context_id'],mode,e['sample_index'])]
            if e['turns'][0]['generation']!=first['generation'] or {k:v for k,v in e['turns'][0].items() if k not in ('generation','sampling_seed')}!=first['row']:
                raise ValueError('continuation first action binding differs')
            records=[r['row'] for r in c['records']]+e['turns']
            calendar_cases.append({'kind':'continuation','model_branch':branch,'mode':mode,'task_id':e['task_id'],'context_id':c['context_id'],
                'sample_index':e['sample_index'],'result':calendar_episode_audit(tm[c['task_id']],records,e['success'],e['artifact'])})
        actions.extend(rows);continuations.extend(episodes)
    recovery=recovery_summary(actions,continuations,cfg['evaluation_branches'])
    if recovery!=read(output/'recovery_comparison.json'):raise ValueError('independent recovery summary differs')
    calendar_summary={}
    for branch in cfg['evaluation_branches']:
        calendar_summary[branch]={}
        for kind in ('complete','continuation'):
            calendar_summary[branch][kind]={}
            for mode in ('greedy','sample'):
                cases=[c for c in calendar_cases if c['model_branch']==branch and c['kind']==kind and c['mode']==mode]
                calendar_summary[branch][kind][mode]={'episodes':len(cases),'environment_successes':sum(c['result']['environment_success'] for c in cases),
                    'calendar_consistent_successes':sum(c['result']['calendar_consistent_success'] for c in cases),
                    'verified_queries':sum(c['result']['verified_query_count'] for c in cases),'calendar_queries_passed':sum(c['result']['calendar_queries_passed'] for c in cases)}
        calendar_summary[branch]['first_action_queries']={mode:{'queries':len([c for c in calendar_first if c['model_branch']==branch and c['mode']==mode]),
            'passed':sum(c['check']['passed'] for c in calendar_first if c['model_branch']==branch and c['mode']==mode)} for mode in ('greedy','sample')}
    paired_continuation={}
    for mode in ('greedy','sample'):
        before={(e['context_id'],e['sample_index']):e for e in continuations if e['model_branch']=='before' and e['mode']==mode}
        after={(e['context_id'],e['sample_index']):e for e in continuations if e['model_branch']=='after' and e['mode']==mode}
        paired_continuation[mode]={'both_success':sum(before[k]['success'] and after[k]['success'] for k in before),
            'both_failure':sum(not before[k]['success'] and not after[k]['success'] for k in before),
            'improved':sum(not before[k]['success'] and after[k]['success'] for k in before),
            'regressed':sum(before[k]['success'] and not after[k]['success'] for k in before)}
    evidence={'episodes_replayed':episode_count,'complete_episode_turns':turns,'first_recovery_actions_replayed':len(actions),
        'recovery_continuations_replayed':len(continuations),'continuation_turns_replayed':continuation_turns,'test_tasks':len(tasks),
        'standard_recovery_contexts':sum(c['scenario'] not in NOVEL_SCENARIOS for c in contexts),'novel_recovery_contexts':sum(c['scenario'] in NOVEL_SCENARIOS for c in contexts),
        'train_test_task_family_disjoint':True,'test_contexts_in_training':0,'training_enabled':False,'optimizer_updates':0}
    summary={'complete':{'greedy':comparisons,'sample':stochastic,'paired_greedy':paired(complete['before']['greedy'],complete['after']['greedy']),
        'paired_sample':paired(complete['before']['sample'],complete['after']['sample'],True)},'recovery':recovery,'paired_continuation':paired_continuation,
        'calendar_validation':calendar_summary,'status':'awaiting_local_review',**evidence}
    calendar={'summary':calendar_summary,'cases':calendar_cases,'first_action_queries':calendar_first}
    return evidence,summary,calendar

def export_counts(output,summary):
    fields=['branch','mode','scenario','actions','productive','expert_tool','evidence_actions','all_evidence_valid','query_required','query_chosen','query_target_verified']
    with (output/'recovery_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for b,modes in summary['recovery'].items():
            for mode,g in modes.items():
                for scenario,s in [('overall',g['first_action']['overall']),*g['first_action']['by_scenario'].items()]:
                    writer.writerow(dict(zip(fields,[b,mode,scenario,s['actions'],s['productive'],s['tool_matches_expert'],s['evidence']['actions'],
                        s['evidence']['all_references_verified'],s['sql']['query_required_actions'],s['sql']['query_chosen_when_required'],s['sql']['target_verified_when_required']])))

def audit(output):
    cfg=verify_inputs(output)
    from transformers import AutoTokenizer
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if (output/'tokenizer'/n).exists() and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved test tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    evidence,summary,calendar=audit_results(output,cfg,tok)
    write(output/'independent_test_summary.json',summary);write(output/'calendar_sql_validation.json',calendar);export_counts(output,summary)
    verify_inputs(output);write(output/'audit.json',{'passed':True,**evidence,'checkpoints_unchanged':True,'calendar_challenges_checked':True})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
