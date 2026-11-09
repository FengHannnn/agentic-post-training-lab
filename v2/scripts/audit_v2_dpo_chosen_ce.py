"""Audit both real optimizer states, CE scaling, exact reference, and full episodes."""
import sys,math,csv
from pathlib import Path
from dpo_chosen_ce_common import ROOT,read,write,load,tasks,contexts,seed_for,result_summary,verify_inputs,training_path,VARIANTS,model_specs
from common import file_hash
from apt_v2.dpo_recovery import SCENARIOS,encode_recovery_pair
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.hard_recovery_sft import recovery_weights
from apt_v2.matched_dpo import replay_recovery
from apt_v2.dpo_test import replay_episode
from apt_v2.recovery_diagnostics import diagnose_action
from apt_v2.independent_recovery_test import calendar_episode_audit
from apt_v2.dpo_training import input_id
from apt_v2.sft import encode_example
from apt_v2.dpo_investigation import build_probes,metrics_from_scores,summarize_pair_metrics,summarize_probe_metrics
from apt_v2.reference_numerics import compare_reference_caches
from apt_v2.dpo_chosen_ce import verify_chosen_ce_updates,numeric_equal

def audit_scores(output,cfg,tok):
    pairs=load(output/'pairs.jsonl');encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ref=read(output/'reference_scores.json');weights=read(output/'weights.json')['weights']
    verify_reference(pairs,encoded,ref,cfg['checkpoints']['parent']['adapter_sha256'])
    probes=build_probes()
    if probes!=load(output/'probes.jsonl'):raise ValueError('deterministic expert probes differ')
    pe=[encode_example(tok,p,cfg['max_length']) for p in probes];summaries={}
    recheck=read(output/'reference_recheck.json');fresh=recheck['fresh_scores']
    verify_reference(pairs,encoded,{'baseline_adapter_sha256':cfg['checkpoints']['parent']['adapter_sha256'],'pair_ids':ref['pair_ids'],'scores':fresh},cfg['checkpoints']['parent']['adapter_sha256'])
    maximum=max(abs(x-y) for a,b in zip(fresh,ref['scores']) for k in ('chosen','rejected') for x,y in zip(a[k],b[k]))
    if maximum!=recheck['maximum_token_logp_difference']:raise ValueError('parent reference recheck differs')
    numerical=compare_reference_caches(dict(ref,scores=fresh),ref,cfg['beta'])
    if not numeric_equal(numerical,recheck['numerics']):raise ValueError('scoring reference precheck differs')
    for branch,spec in cfg['checkpoints'].items():
        saved=read(output/(branch+'-scores.json'))
        if saved['branch']!=branch or saved['adapter_sha256']!=spec['adapter_sha256'] or len(saved['pair_scores'])!=len(encoded):raise ValueError('scored checkpoint identity differs')
        metrics=[]
        for current,r,e in zip(saved['pair_scores'],ref['scores'],encoded):
            if current['input_id']!=input_id(e):raise ValueError('pair score input differs')
            metrics.append(metrics_from_scores(current,r))
        if not numeric_equal(metrics,saved['pair_metrics']):raise ValueError('recomputed pair metrics differ')
        pm=saved['probe_metrics']
        if len(pm)!=len(pe) or any(len(m['token_log_probs'])!=e['response_length'] for m,e in zip(pm,pe)):raise ValueError('probe token lengths differ')
        summary={'pairs':summarize_pair_metrics(pairs,encoded,metrics,ref['scores'],weights),'probes':summarize_probe_metrics(probes,pm)}
        if not numeric_equal(summary,saved['summary']):raise ValueError('checkpoint score summary differs')
        summaries[branch]=summary
    saved=read(output/'scores_summary.json')
    if not numeric_equal(summaries,saved['branches']) or saved['training_enabled'] is not False or saved['optimizer_updates']!=0 or saved['maximum_parent_reference_difference']!=maximum:raise ValueError('score collection summary differs')
    return summaries,maximum

def audit_results(output,cfg,tok):
    ts=tasks();tm={t.id:t for t in ts};cs=contexts();cm={c['context_id']:c for c in cs};complete=[];recoveries=[];actions=[];calendar=[];turns=0
    expected_complete={(t.id,m,i) for t in ts for m in ('greedy','sample') for i in range(1 if m=='greedy' else cfg['sample_eval']['rollouts_per_task'])}
    expected_recovery={(c['context_id'],m,i) for c in cs for m in ('greedy','sample') for i in range(1 if m=='greedy' else cfg['recovery_eval']['samples_per_context'])}
    for b in cfg['evaluation_branches']:
        full=load(output/(b+'-complete-episodes.jsonl'));rs=load(output/(b+'-recovery-episodes.jsonl'));aa=load(output/(b+'-first-actions.jsonl'))
        for rows,key,expected in [(full,lambda e:(e['task_id'],e['mode'],e['rollout_index']),expected_complete),(rs,lambda e:(e['context_id'],e['mode'],e['sample_index']),expected_recovery),(aa,lambda e:(e['context_id'],e['mode'],e['sample_index']),expected_recovery)]:
            if len(rows)!=len(expected) or {key(e) for e in rows}!=expected:raise ValueError('complete distinct evaluation coverage required')
        amap={(e['context_id'],e['mode'],e['sample_index']):e for e in aa}
        for kind,rows in [('complete',full),('recovery',rs)]:
            for e in rows:
                task=tm[e['task_id']];mode=e['mode'];c=None if kind=='complete' else cm[e['context_id']]
                ri=e['rollout_index'] if kind=='complete' else e['sample_index'];seed=seed_for(cfg,task,mode,ri,c)
                temperature=0 if mode=='greedy' else cfg['sample_eval' if kind=='complete' else 'recovery_eval']['temperature']
                if e['model_branch']!=b or e['split']!='development' or e['sample_seed']!=seed or e['temperature']!=temperature or any(r['sampling_seed']!=seed+i for i,r in enumerate(e['turns'])):raise ValueError('paired evaluation identity differs')
                native_mode='greedy' if mode=='greedy' else 'dpo_sample'
                turns+=replay_episode(task,e,tok,native_mode) if kind=='complete' else replay_recovery(task,c,e,tok,native_mode)
                for row in e['turns']:
                    g=row['generation'];ids=g['response_token_ids']
                    if len(ids)>cfg['eval']['max_new_tokens'] or len(g['prompt_token_ids'])+cfg['eval']['max_new_tokens']>cfg['eval']['max_model_len'] or (g['finish_reason']=='stop' and ids[-1]!=tok.eos_token_id):raise ValueError('native token budget or EOS differs')
                records=e['turns'] if c is None else [r['row'] for r in c['records']]+e['turns']
                calendar.append({'kind':kind,'branch':b,'mode':mode,'task_id':task.id,'index':ri,'context_id':e.get('context_id'),
                    'result':calendar_episode_audit(task,records,e['success'],e['artifact'],allowed_split='development')})
                if c is not None:
                    a=amap[(c['context_id'],mode,ri)];g=e['turns'][0]['generation'];actual,d=diagnose_action(task,c,g['text'])
                    expected={'model_branch':b,'mode':mode,'task_id':task.id,'family_id':task.family_id,'split':'development','context_id':c['context_id'],
                        'scenario':c['scenario'],'sample_index':ri,'sample_seed':seed,'temperature':temperature,'generation':g,'row':actual,'diagnostic':d}
                    if a!=expected or actual!={k:v for k,v in e['turns'][0].items() if k not in ('generation','sampling_seed')}:raise ValueError('first action continuation binding differs')
        complete.extend(full);recoveries.extend(rs);actions.extend(aa)
    summary=result_summary(complete,recoveries,actions,cfg)
    if summary!=read(output/'comparison_summary.json'):raise ValueError('paired summary differs')
    evidence={'complete_episodes_replayed':len(complete),'recovery_episodes_replayed':len(recoveries),'first_actions_replayed':len(actions),'turns_replayed':turns,'test_split_used':False}
    return summary,calendar,evidence


def audit_training(output,cfg,tok):
    import torch
    pairs=load(output/'pairs.jsonl');weights,ws=recovery_weights(pairs,4);encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ref=read(output/'reference_scores.json')
    if file_hash(output/'reference_scores.json')!=cfg['reference_scores_sha256'] or read(output/'weights.json')!={'weights':weights,'summary':ws}:raise ValueError('canonical training inputs differ')
    verify_reference(pairs,encoded,ref,cfg['adapter_sha256']);initial={};evidence={}
    for branch in VARIANTS:
        child=training_path(output,branch);final=read(child/'final/complete.json');train=read(child/'training_summary.json')
        if file_hash(child/'final/state.pt')!=final['state_sha256']:raise ValueError('real optimizer state hash differs')
        state=torch.load(child/'final/state.pt',map_location='cpu',weights_only=True);verify_chosen_ce_updates(cfg,branch,final,train,state)
        if file_hash(child/'reference_scores.json')!=cfg['reference_scores_sha256'] or train['reference_scores_sha256']!=cfg['reference_scores_sha256'] or load(child/'training_steps.jsonl')!=train['history'] or train['weight_summary']!=ws or train['adapter_tensors_changed']<=0:raise ValueError('actual paired optimization evidence differs')
        if any(g['lr']!=cfg['learning_rate'] or g['weight_decay']!=0 for g in state['optimizer']['param_groups']):raise ValueError('paired fresh Adam hyperparameters differ')
        if read(child/'execution_environment.json')!=read(output/'execution_environment.json'):raise ValueError('paired training execution environments differ')
        pre=read(child/'reference_precheck.json');comparison=compare_reference_caches(dict(ref,scores=pre['fresh_scores']),ref,cfg['beta'])
        if not numeric_equal(comparison,pre['numerics']) or pre['canonical_sha256']!=cfg['reference_scores_sha256'] or pre['before_any_optimizer_update'] is not True:raise ValueError('pre-update reference verification differs')
        initial[branch]=dict(ref,scores=pre['fresh_scores'])
        for name in ['before_pair_scores.json','after_pair_scores.json']:
            scored=read(child/name);ms=scored['metrics']
            if len(ms)!=169 or scored['pairs']!=169:raise ValueError('paired training score count differs')
            for k,v in scored['means'].items():
                if not math.isfinite(v) or abs(sum(m[k] for m in ms)/169-v)>1e-8:raise ValueError('training score summary differs')
            if abs(sum(w*m['loss'] for w,m in zip(weights,ms))/169-scored['weighted_mean_loss'])>1e-8:raise ValueError('weighted training score differs')
            for m,e,r in zip(ms,encoded,ref['scores']):
                z=-cfg['beta']*m['relative_margin'];loss=max(0,z)+math.log1p(math.exp(-abs(z)));margin=(m['chosen_mean_logp']*len(e['chosen'])-m['rejected_mean_logp']*len(e['rejected']))-(sum(r['chosen'])-sum(r['rejected']))
                if not all(math.isfinite(v) for v in m.values()) or abs(loss-m['loss'])>1e-4 or abs(margin-m['relative_margin'])>.002:raise ValueError('DPO score formula differs')
            if name.startswith('before') and (abs(scored['means']['loss']-math.log(2))>5e-5 or max(abs(m['relative_margin']) for m in ms)>.002):raise ValueError('initial canonical-reference score differs')
        for step in cfg['checkpoint_steps']:
            c=read(child/f'checkpoint-{step:04}'/'complete.json');coefficient=cfg['training_variants'][branch]['chosen_ce_coefficient']
            if c['steps']!=step or c['cursor']!=step//43*169 or c['epochs_completed']!=step//43 or c['total_sft_updates']!=255 or c['total_dpo_updates']!=11+step or c['branch']!=branch or c['chosen_ce_coefficient']!=coefficient or c['auxiliary_chosen_ce_updates']!=(step if coefficient else 0):raise ValueError('paired intermediate ancestry differs')
        evidence[branch]={'optimizer_updates':258,'pair_exposures':1014,'chosen_ce_coefficient':cfg['training_variants'][branch]['chosen_ce_coefficient'],'chosen_ce_pair_exposures':train['chosen_ce_pair_exposures'],'reference_scores_sha256':cfg['reference_scores_sha256']}
    return evidence,compare_reference_caches(initial['weighted_control'],initial['dpo_ce'],cfg['beta'])

def audit(output):
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);identity=read(output/'model_identity.json')
    for n,h in identity['files_sha256'].items():
        if n.endswith(('.json','.txt','.jinja')) and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    training,initial=audit_training(output,cfg,tok);specs=model_specs(cfg,output)
    if specs!=read(output/'score_model_specs.json'):raise ValueError('five scored models differ')
    scores,maximum=audit_scores(output,dict(cfg,checkpoints=specs),tok);summary,calendar,evidence=audit_results(output,cfg,tok)
    write(output/'training_audit.json',{'branches':training,'paired_initial_numerics':initial,'total_optimizer_updates':516})
    write(output/'score_comparison.json',{'branches':scores,'reference_scores_sha256':cfg['reference_scores_sha256'],'maximum_fresh_reference_difference':maximum})
    write(output/'evaluation_summary.json',dict(summary,status='awaiting_local_review',**evidence));write(output/'calendar_sql_validation.json',calendar)
    with (output/'results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['branch','mode','kind','episodes','successes'])
        for b,modes in summary['branches'].items():
            for mode,s in modes.items():
                for kind in ('complete','recovery'):w.writerow([b,mode,kind,s[kind]['episodes'],s[kind]['successes']])
    verify_inputs(output);write(output/'audit.json',dict(passed=True,optimizer_updates_this_job=516,training_branches=2,scored_models=5,probes_per_model=238,reference_inputs_verified=169,canonical_reference_sha256=cfg['reference_scores_sha256'],**evidence))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
