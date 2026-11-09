"""Replay E1 samples and verify every score, token binding and checkpoint label."""
import sys,csv,math
from pathlib import Path
from dpo_investigation_common import ROOT,read,write,load,jsonl,verify_inputs,seed_for
from apt_v2.dpo_investigation import investigation_tasks,investigation_contexts,build_probes,metrics_from_scores,summarize_pair_metrics,summarize_probe_metrics,evaluation_summary,digest
from apt_v2.dpo_recovery import encode_recovery_pair,SCENARIOS
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.dpo_training import input_id
from apt_v2.dpo_test import replay_episode
from apt_v2.matched_dpo import replay_recovery
from apt_v2.sft import encode_example
from apt_v2.recovery_diagnostics import diagnose_action,summarize
from apt_v2.independent_recovery_test import calendar_episode_audit
from common import file_hash

def audit_scores(output,cfg,tok):
    pairs=load(output/'pairs.jsonl');encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ref=read(output/'reference_scores.json');weights=read(output/'weights.json')['weights']
    verify_reference(pairs,encoded,ref,cfg['checkpoints']['parent']['adapter_sha256'])
    probes=build_probes()
    if probes!=load(output/'probes.jsonl'):raise ValueError('deterministic expert probes differ')
    pe=[encode_example(tok,p,cfg['max_length']) for p in probes];summaries={}
    recheck=read(output/'reference_recheck.json');fresh=recheck['fresh_scores']
    verify_reference(pairs,encoded,{'baseline_adapter_sha256':cfg['checkpoints']['parent']['adapter_sha256'],'pair_ids':ref['pair_ids'],'scores':fresh},cfg['checkpoints']['parent']['adapter_sha256'])
    maximum=max(abs(x-y) for a,b in zip(fresh,ref['scores']) for k in ('chosen','rejected') for x,y in zip(a[k],b[k]))
    if maximum!=recheck['maximum_token_logp_difference'] or maximum>1e-4:raise ValueError('parent reference recheck differs')
    for branch,spec in cfg['checkpoints'].items():
        saved=read(output/(branch+'-scores.json'))
        if saved['branch']!=branch or saved['adapter_sha256']!=spec['adapter_sha256'] or len(saved['pair_scores'])!=len(encoded):raise ValueError('scored checkpoint identity differs')
        metrics=[]
        for current,r,e in zip(saved['pair_scores'],ref['scores'],encoded):
            if current['input_id']!=input_id(e):raise ValueError('pair score input differs')
            metrics.append(metrics_from_scores(current,r))
        if metrics!=saved['pair_metrics']:raise ValueError('recomputed pair metrics differ')
        pm=saved['probe_metrics']
        if len(pm)!=len(pe) or any(len(m['token_log_probs'])!=e['response_length'] for m,e in zip(pm,pe)):raise ValueError('probe token lengths differ')
        summary={'pairs':summarize_pair_metrics(pairs,encoded,metrics,ref['scores'],weights),'probes':summarize_probe_metrics(probes,pm)}
        if summary!=saved['summary']:raise ValueError('checkpoint score summary differs')
        summaries[branch]=summary
    saved=read(output/'scores_summary.json')
    if summaries!=saved['branches'] or saved['training_enabled'] is not False or saved['optimizer_updates']!=0 or saved['maximum_parent_reference_difference']!=maximum:raise ValueError('score collection summary differs')
    return summaries,maximum

def audit_evaluations(output,cfg,tok):
    ts=investigation_tasks();cs=investigation_contexts();tm={t.id:t for t in ts};cm={c['context_id']:c for c in cs};complete=[];recovery=[];actions=[];calendar=[];turns=0
    expected_complete={(t.id,m,i) for t in ts for m in ('greedy','sample') for i in range(1 if m=='greedy' else cfg['sample_eval']['rollouts_per_task'])}
    expected_recovery={(c['context_id'],'greedy',0) for c in cs}
    if load(output/'recovery_contexts.jsonl')!=cs:raise ValueError('recovery contexts differ')
    for branch,spec in cfg['checkpoints'].items():
        for kind,expected in [('complete',expected_complete),('recovery',expected_recovery)]:
            es=load(output/(branch+'-'+kind+'-episodes.jsonl'));key=(lambda e:(e['task_id'],e['mode'],e['rollout_index'])) if kind=='complete' else (lambda e:(e['context_id'],e['mode'],e['sample_index']))
            if len(es)!=len(expected) or {key(e) for e in es}!=expected:raise ValueError('complete distinct E1 coverage required')
            for e in es:
                task=tm[e['task_id']];mode=e['mode'];c=None if kind=='complete' else cm[e['context_id']];ri=e['rollout_index'] if c is None else e['sample_index'];seed=seed_for(cfg,task,mode,ri,c)
                if e['model_branch']!=branch or e['adapter_sha256']!=spec['adapter_sha256'] or e['sample_seed']!=seed or e['temperature']!=(0 if mode=='greedy' else cfg['sample_eval']['temperature']) or any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):raise ValueError('checkpoint branch or seed differs')
                nm='greedy' if mode=='greedy' else 'dpo_sample'
                turns+=replay_episode(task,e,tok,nm) if c is None else replay_recovery(task,c,e,tok,nm)
                for row in e['turns']:
                    g=row['generation'];ids=g['response_token_ids']
                    if len(ids)>cfg['eval']['max_new_tokens'] or len(g['prompt_token_ids'])+cfg['eval']['max_new_tokens']>cfg['eval']['max_model_len'] or (g['finish_reason']=='stop' and ids[-1]!=tok.eos_token_id):raise ValueError('native generation budget differs')
                records=e['turns'] if c is None else [r['row'] for r in c['records']]+e['turns']
                calendar.append({'branch':branch,'kind':kind,'mode':mode,'task_id':task.id,'context_id':e.get('context_id'),'index':ri,'result':calendar_episode_audit(task,records,e['success'],e['artifact'],allowed_split='development')})
                if c is not None:
                    actual,d=diagnose_action(task,c,e['turns'][0]['generation']['text'])
                    if actual!={k:v for k,v in e['turns'][0].items() if k not in ('generation','sampling_seed')}:raise ValueError('recovery first-action binding differs')
                    actions.append({'model_branch':branch,'context_id':c['context_id'],'scenario':c['scenario'],'diagnostic':d})
            (complete if kind=='complete' else recovery).extend(es)
    summary=evaluation_summary(complete,recovery,cfg['checkpoints'])
    if summary!=read(output/'comparison_summary.json'):raise ValueError('paired E1 summary differs')
    first={b:summarize([r for r in actions if r['model_branch']==b]) for b in cfg['checkpoints']}
    evidence={'complete_episodes_replayed':len(complete),'recovery_episodes_replayed':len(recovery),'first_actions_replayed':len(actions),'turns_replayed':turns}
    return summary,calendar,first,evidence

def onset_report(cfg,scores,summary):
    parent=summary['branches']['parent'];rows=[]
    for branch,spec in sorted(cfg['checkpoints'].items(),key=lambda item:item[1]['new_dpo_steps']):
        if not spec['new_dpo_steps']:continue
        s=summary['branches'][branch];p=scores[branch]['pairs']['by_scenario']['stale_evidence_repair'];pp=scores['parent']['pairs']['by_scenario']['stale_evidence_repair']
        rows.append({'branch':branch,'steps':spec['new_dpo_steps'],'task_greedy':s['complete']['greedy']['successes'],'task_sample':s['complete']['sample']['successes'],
            'recovery_greedy':s['recovery_greedy']['successes'],'stale_chosen_ce':p['chosen_ce'],'stale_ce_ratio_to_parent':p['chosen_ce']/pp['chosen_ce'],
            'task_greedy_below_parent':s['complete']['greedy']['successes']<parent['complete']['greedy']['successes'],
            'task_sample_below_parent':s['complete']['sample']['successes']<parent['complete']['sample']['successes'],
            'recovery_greedy_below_parent':s['recovery_greedy']['successes']<parent['recovery_greedy']['successes']})
    first=lambda key:next((r['steps'] for r in rows if r[key]),None)
    return {'milestones':rows,'first_task_greedy_regression_step':first('task_greedy_below_parent'),'first_task_sample_regression_step':first('task_sample_below_parent'),
        'first_recovery_greedy_regression_step':first('recovery_greedy_below_parent'),'checkpoint_selection':'diagnosis_only_no_promotion','C1_status':'failed_pending_causal_experiments'}

def audit(output):
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if n.endswith(('.json','.txt','.jinja')) and file_hash(output/'tokenizer'/n)!=h:raise ValueError('diagnostic tokenizer differs')
    scores,maximum=audit_scores(output,cfg,tok);summary,calendar,first,evidence=audit_evaluations(output,cfg,tok)
    result={'evaluation':summary,'score_summaries':scores,'first_action':first,'onset':onset_report(cfg,scores,summary),'training_enabled':False,'optimizer_updates':0,'test_split_used':False,**evidence}
    write(output/'diagnosis_summary.json',result);write(output/'calendar_sql_validation.json',calendar)
    with (output/'checkpoint_counts.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['branch','new_dpo_steps','task_greedy','task_sample','recovery_greedy','stale_chosen_ce'])
        for b,spec in cfg['checkpoints'].items():
            s=summary['branches'][b];w.writerow([b,spec['new_dpo_steps'],s['complete']['greedy']['successes'],s['complete']['sample']['successes'],s['recovery_greedy']['successes'],scores[b]['pairs']['by_scenario']['stale_evidence_repair']['chosen_ce']])
    verify_inputs(output);write(output/'audit.json',{'passed':True,'training_enabled':False,'optimizer_updates':0,'test_split_used':False,'scored_checkpoints':len(cfg['checkpoints']),'reference_pairs_verified':169,'maximum_parent_reference_difference':maximum,**evidence})

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
