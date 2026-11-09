import sys,json,math,statistics,collections,hashlib,csv
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
out=Path(sys.argv[1]);dest=Path(sys.argv[2]);project=Path(sys.argv[3]);cfg=json.loads((out/'config.json').read_text(encoding='utf8'))
read=lambda p:json.loads(p.read_text(encoding='utf8'))
load=lambda p:[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
pairs=load(out/'pairs.jsonl');before=read(out/'before_pair_scores.json')['metrics'];after=read(out/'after_pair_scores.json')['metrics'];weights=read(out/'weights.json')['weights'];refs=read(out/'reference_scores.json')['scores']
sys.path[:0]=[str(out/'source/scripts'),str(out/'source/src')]
from transformers import AutoTokenizer
from apt_v2.dpo_recovery import encode_recovery_pair
from apt_v2.hard_recovery_sft import HARD_SCENARIOS
from apt_v2.dpo import action_from_text
from apt_v2.chosen_sft import chosen_examples,encode_chosen,data_sha256
from apt_v2.dpo_training import input_id
from safetensors.torch import load_file
import torch
tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True,trust_remote_code=False)
encoded=[encode_recovery_pair(tok,p,2048) for p in pairs]
comp=project/'artifacts/reviews'/cfg['comparison_run']/'extracted'/cfg['comparison_run'];sft=read(comp/'after_chosen_scores.json');ce=encode_chosen(tok,chosen_examples(pairs),2048)
assert all(m['input_id']==data_sha256(e) for m,e in zip(sft['metrics'],ce))
rows=[]
for i,(p,b,a,e,w) in enumerate(zip(pairs,before,after,encoded,weights)):
    dc=(a['chosen_mean_logp']-b['chosen_mean_logp'])*len(e['chosen']);dr=(a['rejected_mean_logp']-b['rejected_mean_logp'])*len(e['rejected'])
    rows.append({'pair_id':p['pair_id'],'scenario':p.get('scenario','original_sql'),'rejected_source':p['rejected_source'],'chosen_tool':action_from_text(p['chosen']).get('tool'),
        'hard':p.get('scenario') in HARD_SCENARIOS,'weight':w,'chosen_tokens':len(e['chosen']),'rejected_tokens':len(e['rejected']),
        'chosen_before':b['chosen_mean_logp'],'chosen_after':a['chosen_mean_logp'],'rejected_before':b['rejected_mean_logp'],'rejected_after':a['rejected_mean_logp'],
        'delta_chosen_sequence_logp':dc,'delta_rejected_sequence_logp':dr,'relative_margin':a['relative_margin'],'loss_before':b['loss'],'loss_after':a['loss']})
def score_summary(rs):
    mean=lambda k:statistics.mean(r[k] for r in rs)
    return {'pairs':len(rs),'chosen_mean_logp_before':mean('chosen_before'),'chosen_mean_logp_after':mean('chosen_after'),'rejected_mean_logp_before':mean('rejected_before'),'rejected_mean_logp_after':mean('rejected_after'),
        'mean_delta_chosen_sequence_logp':mean('delta_chosen_sequence_logp'),'mean_delta_rejected_sequence_logp':mean('delta_rejected_sequence_logp'),'mean_relative_margin':mean('relative_margin'),
        'chosen_probability_increased':sum(r['chosen_after']>r['chosen_before'] for r in rs),'chosen_probability_decreased':sum(r['chosen_after']<r['chosen_before'] for r in rs),
        'rejected_probability_decreased':sum(r['rejected_after']<r['rejected_before'] for r in rs),'positive_relative_margin':sum(r['relative_margin']>0 for r in rs),
        'loss_before':mean('loss_before'),'loss_after':mean('loss_after'),'weighted_loss_after':sum(r['weight']*r['loss_after'] for r in rs)/sum(r['weight'] for r in rs)}
prob={'overall':score_summary(rows),'by_scenario':{s:score_summary([r for r in rows if r['scenario']==s]) for s in dict.fromkeys(r['scenario'] for r in rows)},
    'by_chosen_tool':{s:score_summary([r for r in rows if r['chosen_tool']==s]) for s in dict.fromkeys(r['chosen_tool'] for r in rows)},'by_hard':{str(s):score_summary([r for r in rows if r['hard']==s]) for s in (True,False)},
    'matched_sft_chosen_ce':{'before':read(comp/'before_chosen_scores.json')['mean_loss'],'after':sft['mean_loss'],'weighted_after':sft['mean_weighted_loss'],'encoded_inputs_verified':169}}
history=load(out/'training_steps.jsonl');epochs=[]
for ep in range(1,7):
    hs=[h for h in history if h['epoch']==ep];sizes=[h['epoch_pairs_seen']-(hs[j-1]['epoch_pairs_seen'] if j else 0) for j,h in enumerate(hs)]
    epochs.append({'epoch':ep,'steps':len(hs),'exposures':sum(sizes),'online_loss_exposure_mean':sum(h['loss']*n for h,n in zip(hs,sizes))/sum(sizes),'grad_norm_mean':statistics.mean(h['grad_norm'] for h in hs),'grad_norm_max':max(h['grad_norm'] for h in hs),'clipped_steps':sum(h['grad_norm']>1 for h in hs)})
def behavior(es):
    turns=[r for e in es for r in e['turns']];tools=collections.Counter(r['action'].get('tool','<invalid>') for r in turns);finishes=collections.Counter(r['generation']['finish_reason'] for r in turns)
    queried=collections.Counter(m for r in turns for m in r['observation'].get('verified_metrics',[]) if r['action'].get('tool')=='query_sql')
    exact_repeat=0;repeat_sql=0;no_progress=0;check_fail=0
    for e in es:
        seen=set()
        for r in e['turns']:
            action=json.dumps(r['action'],sort_keys=True,ensure_ascii=False)
            if action in seen:exact_repeat+=1
            if action in seen and r['action'].get('tool')=='query_sql':repeat_sql+=1
            seen.add(action)
            if r['action'].get('tool') in ('calculate_change','query_sql') and not r['new_verified_metrics']:no_progress+=1
            if r['action'].get('tool')=='check_result' and r['observation'].get('passed') is False:check_fail+=1
    def metrics(e):return {m for r in e['turns'] for m in r['new_verified_metrics']}
    return {'episodes':len(es),'successes':sum(e['success'] for e in es),'turns':len(turns),'tools':dict(tools),'finish_reasons':dict(finishes),
        'verified_query_metrics':dict(queried),'episodes_with_no_new_metric':sum(not metrics(e) for e in es),'episodes_with_both_monthly_newly_verified':sum({'current_net_cents','prior_net_cents'}<=metrics(e) for e in es),
        'episodes_with_change_newly_verified':sum('change_cents' in metrics(e) for e in es),'episodes_with_calculate_change':sum(any(r['action'].get('tool')=='calculate_change' for r in e['turns']) for e in es),
        'episodes_with_submit':sum(any(r['action'].get('tool')=='submit' for r in e['turns']) for e in es),'within_continuation_repeated_actions':exact_repeat,'within_continuation_repeated_sql':repeat_sql,'query_or_calculate_without_new_metric':no_progress,'failed_checks':check_fail,
        'tool_errors':dict(collections.Counter(r['tool_error'] for r in turns if r['tool_error'])),'termination':dict(collections.Counter(e.get('termination',e['turns'][-1]['termination']) for e in es)),
        'response_tokens':sum(len(r['generation']['response_token_ids']) for r in turns)}
beh={};cases=[];first={};calendar=read(out/'calendar_sql_validation.json');cal_summary={}
for branch in cfg['evaluation_branches']:
    beh[branch]={};first[branch]={};cal_summary[branch]={}
    aa=load(out/(branch+'-first-actions.jsonl'))
    for mode in ('greedy','sample'):
        first[branch][mode]={'actions':sum(a['mode']==mode for a in aa),'productive':sum(a['mode']==mode and a['diagnostic']['productive'] for a in aa),'tools':dict(collections.Counter(a['diagnostic']['actual_tool'] for a in aa if a['mode']==mode)),
            'repeated_previous_action':sum(a['mode']==mode and a['diagnostic']['repeated_previous_action'] for a in aa),'by_scenario':{s:{'actions':sum(a['mode']==mode and a['scenario']==s for a in aa),'productive':sum(a['mode']==mode and a['scenario']==s and a['diagnostic']['productive'] for a in aa)} for s in dict.fromkeys(a['scenario'] for a in aa)}}
        beh[branch][mode]={};cal_summary[branch][mode]={}
        for kind in ('complete','recovery'):
            es=[e for e in load(out/(branch+'-'+kind+'-episodes.jsonl')) if e['mode']==mode];beh[branch][mode][kind]=behavior(es)
            subset=[c for c in calendar if c['branch']==branch and c['mode']==mode and c['kind']==kind]
            cal_summary[branch][mode][kind]={'episodes':len(subset),'environment_successes':sum(c['result']['environment_success'] for c in subset),'calendar_consistent_successes':sum(c['result']['calendar_consistent_success'] for c in subset),'queries':sum(c['result']['verified_query_count'] for c in subset),'queries_passed':sum(c['result']['calendar_queries_passed'] for c in subset)}
            for e in es:
                cases.append({'branch':branch,'mode':mode,'kind':kind,'task_id':e['task_id'],'scenario':e.get('scenario'),'index':e.get('rollout_index',e.get('sample_index')),'success':e['success'],'turns':len(e['turns']),
                    'trace':[{'turn':r['turn'],'action':r['action'],'verified_metrics':r['observation'].get('verified_metrics',[]),'new_verified_metrics':r['new_verified_metrics'],'checks':r['observation'].get('checks'),'error':r['tool_error'],'finish_reason':r['generation']['finish_reason']} for r in e['turns']]})
parent=project/'artifacts/reviews'/cfg['initial_run']/'extracted'/cfg['initial_run'];a=load_file(parent/'final/adapter/adapter_model.safetensors');b=load_file(out/'final/adapter/adapter_model.safetensors')
assert a.keys()==b.keys();delta=[]
for n in a:
    assert a[n].shape==b[n].shape and torch.isfinite(b[n]).all()
    delta.append({'name':n,'parent_norm':a[n].norm().item(),'final_norm':b[n].norm().item(),'delta_norm':(b[n]-a[n]).norm().item(),'changed':not torch.equal(a[n],b[n])})
contingency={}
for branch in cfg['evaluation_branches']:
    aa=load(out/(branch+'-first-actions.jsonl'));es=load(out/(branch+'-recovery-episodes.jsonl'));em={(e['context_id'],e['mode'],e['sample_index']):e for e in es};contingency[branch]={}
    for mode in ('greedy','sample'):
        counts={'productive_and_success':0,'productive_and_failure':0,'unproductive_and_success':0,'unproductive_and_failure':0}
        for r in aa:
            if r['mode']!=mode:continue
            e=em[(r['context_id'],mode,r['sample_index'])];counts[('productive' if r['diagnostic']['productive'] else 'unproductive')+'_and_'+('success' if e['success'] else 'failure')]+=1
        contingency[branch][mode]=counts
analysis={'first_action_to_final':contingency,'probability':prob,'training_epochs':epochs,'behavior':beh,'first_action':first,'calendar_summary':cal_summary,'adapter':{'tensors':len(delta),'all_finite':True,'changed':sum(d['changed'] for d in delta),'l2_delta':math.sqrt(sum(d['delta_norm']**2 for d in delta))},'case_count':len(cases)}
(dest/'behavior_analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf8');(dest/'recovery_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf8')
with (dest/'pair_probability_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'probability':prob,'training_epochs':epochs,'dpo_behavior':beh['dpo'],'first_action':first,'calendar_summary':cal_summary,'adapter':analysis['adapter']},ensure_ascii=False,indent=2))
