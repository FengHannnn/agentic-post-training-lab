"""Strict early-checkpoint comparison replay with small same-input quality checks."""
import sys,csv
from pathlib import Path
from dpo_early86_common import ROOT,read,write,load,tasks,contexts,seed_for,result_summary,verify_inputs
from apt_v2.matched_dpo import replay_recovery
from apt_v2.dpo_test import replay_episode
from apt_v2.recovery_diagnostics import diagnose_action
from apt_v2.independent_recovery_test import calendar_episode_audit
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



def audit_quality(output,cfg,tok):
    from apt_v2.dpo_recovery_training import validate_generation
    evidence={};selected=contexts()[:cfg['protocol']['quality_contexts_per_model']];cm={c['context_id']:c for c in selected};tm={t.id:t for t in tasks()}
    for b in cfg['evaluation_branches']:
        rows=load(output/(b+'-quality.jsonl'))
        if len(rows)!=len(cm) or {r['context_id'] for r in rows}!=cm.keys():raise ValueError('fixed small quality coverage differs')
        mismatches=[];maximum=0.
        for row in rows:
            c=cm[row['context_id']];task=tm[c['task_id']];seed=seed_for(cfg,task,'greedy',0,c)
            if {k:row[k] for k in ('model_branch','task_id','context_id','scenario','split','seed')}!={'model_branch':b,'task_id':task.id,'context_id':c['context_id'],'scenario':c['scenario'],'split':'development','seed':seed} or len(row['generations'])!=2:raise ValueError('quality identity differs')
            for g in row['generations']:
                validate_generation(tok,c['messages'],g,'greedy');ids=g['response_token_ids']
                if len(ids)>cfg['eval']['max_new_tokens'] or len(g['prompt_token_ids'])+cfg['eval']['max_new_tokens']>cfg['eval']['max_model_len'] or (g['finish_reason']=='stop' and ids[-1]!=tok.eos_token_id):raise ValueError('quality native token budget differs')
            left,right=row['generations']
            if left['response_token_ids']!=right['response_token_ids']:mismatches.append(c['context_id'])
            else:maximum=max(maximum,max(abs(a-b) for a,b in zip(left['raw_token_log_probs'],right['raw_token_log_probs'])))
        evidence[b]={'contexts':len(rows),'generated_actions':2*len(rows),'identical_response_contexts':len(rows)-len(mismatches),'mismatches':mismatches,'maximum_raw_logp_difference_on_aligned_actions':maximum}
    return {'passed':all(not v['mismatches'] for v in evidence.values()),'branches':evidence}

def performance_comparison(summary):
    base=summary['branches']['weighted_sft'];candidate=summary['branches']['dpo_0086'];result={}
    for mode in ('greedy','sample'):
        result[mode]={}
        for kind in ('complete','recovery'):
            a=base[mode][kind];b=candidate[mode][kind]
            result[mode][kind]={'SFT_successes':a['successes'],'DPO86_successes':b['successes'],'episodes_per_model':a['episodes'],'DPO_minus_SFT_successes':b['successes']-a['successes'],'DPO_minus_SFT_percentage_points':100*(b['successes']-a['successes'])/a['episodes'],'paired':summary['paired']['weighted_sft_to_dpo_0086'][mode][kind]}
    return {'primary':'sample.complete','secondary':'sample.recovery','results':result,'performance_acceptance':'awaiting_user_review_of_similarity','candidate_fixed_before_run':True,'training_enabled':False,'optimizer_updates':0}

def audit(output):
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False);complete=[];recovery=[];actions=[]
    for b in cfg['evaluation_branches']:
        record=read(output/(b+'-execution.json'))
        if record!={'pid':record['pid'],'branch':b,'one_adapter_in_this_process':True,'max_loras':1,'lora_name':'fixed_adapter','lora_int_id':1,'training_enabled':False,'optimizer_updates':0}:raise ValueError('isolated constant-position execution differs')
        complete+=load(output/(b+'-complete-episodes.jsonl'));recovery+=load(output/(b+'-recovery-episodes.jsonl'));actions+=load(output/(b+'-first-actions.jsonl'))
    write(output/'comparison_summary.json',result_summary(complete,recovery,actions,cfg));summary,calendar,evidence=audit_results(output,cfg,tok);quality=audit_quality(output,cfg,tok)
    write(output/'quality_checks.json',quality);write(output/'evaluation_summary.json',dict(summary,**evidence));write(output/'performance_comparison.json',performance_comparison(summary));write(output/'calendar_sql_validation.json',calendar)
    with (output/'results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['branch','mode','kind','episodes','successes'])
        for b,ms in summary['branches'].items():
            for mode,s in ms.items():
                for kind in ('complete','recovery'):writer.writerow([b,mode,kind,s[kind]['episodes'],s[kind]['successes']])
    verify_inputs(output);write(output/'audit.json',dict(passed=quality['passed'],training_enabled=False,optimizer_updates=0,small_quality_checks_passed=quality['passed'],**evidence))
    if not quality['passed']:raise ValueError('fixed-input greedy consistency check failed; inspect quality_checks.json')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
