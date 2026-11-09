"""Audit weighted updates, native complete recoveries and paired development results."""
import sys,math,csv
from pathlib import Path
from matched_dpo_common import ROOT,read,write,load,tasks,contexts,seed_for,result_summary
from evaluate_v2_matched_dpo import verify_inputs
from common import file_hash,verify_files
from apt_v2.dpo_recovery import SCENARIOS,encode_recovery_pair
from apt_v2.dpo_recovery_training import verify_reference
from apt_v2.hard_recovery_sft import recovery_weights
from apt_v2.matched_dpo import verify_matched_updates,replay_recovery
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

def audit(output):
    import torch
    from transformers import AutoTokenizer
    cfg=verify_inputs(output);pairs=load(output/'pairs.jsonl');weights,ws=recovery_weights(pairs,cfg['hard_weight'])
    if read(output/'weights.json')!={'weights':weights,'summary':ws}:raise ValueError('normalized weights differ')
    identity=read(output/'model_identity.json')
    if identity!=read(ROOT/'artifacts'/cfg['initial_run']/'model_identity.json'):raise ValueError('training base identity differs')
    for n,h in identity['files_sha256'].items():
        if n.endswith(('.json','.txt','.jinja')) and file_hash(output/'tokenizer'/n)!=h:raise ValueError('saved tokenizer differs')
    tok=AutoTokenizer.from_pretrained(output/'tokenizer',local_files_only=True,trust_remote_code=False)
    encoded=[encode_recovery_pair(tok,p,cfg['max_length']) for p in pairs];ref=read(output/'reference_scores.json')
    verify_reference(pairs,encoded,ref,cfg['adapter_sha256'])
    for name in ('before_pair_scores.json','after_pair_scores.json'):
        scores=read(output/name);ms=scores['metrics']
        if scores['pairs']!=len(pairs) or len(ms)!=len(pairs):raise ValueError('pair score count differs')
        for k,v in scores['means'].items():
            if not math.isfinite(v) or abs(sum(m[k] for m in ms)/len(ms)-v)>1e-8:raise ValueError('pair metric mean differs')
        if abs(sum(w*m['loss'] for w,m in zip(weights,ms))/len(ms)-scores['weighted_mean_loss'])>1e-8:raise ValueError('weighted pair mean differs')
        for m,e,r in zip(ms,encoded,ref['scores']):
            margin=(m['chosen_mean_logp']*len(e['chosen'])-m['rejected_mean_logp']*len(e['rejected']))-(sum(r['chosen'])-sum(r['rejected']))
            z=-cfg['beta']*m['relative_margin'];softplus=max(0,z)+math.log1p(math.exp(-abs(z)))
            if not all(math.isfinite(v) for v in m.values()) or abs(softplus-m['loss'])>1e-4 or abs(margin-m['relative_margin'])>.002:raise ValueError('DPO objective or sequence margin differs')
        if name.startswith('before') and (abs(scores['means']['loss']-math.log(2))>1e-5 or max(abs(m['relative_margin']) for m in ms)>1e-4):raise ValueError('initial reference differs')
    final=read(output/'final/complete.json');train=read(output/'training_summary.json')
    if file_hash(output/'final/state.pt')!=final['state_sha256']:raise ValueError('optimizer state hash differs')
    state=torch.load(output/'final/state.pt',map_location='cpu',weights_only=True);verify_matched_updates(cfg,final,train,state)
    if train['reference_scores_sha256']!=file_hash(output/'reference_scores.json'):raise ValueError('reference cache binding differs')
    if any(g['lr']!=cfg['learning_rate'] or g['weight_decay']!=0 for g in state['optimizer']['param_groups']):raise ValueError('fresh optimizer hyperparameters differ')
    if train['adapter_tensors_changed']<=0 or train['weight_summary']!=ws or load(output/'training_steps.jsonl')!=train['history']:raise ValueError('real weighted training evidence differs')
    for step in cfg['checkpoint_steps']:
        c=read(output/f'checkpoint-{step:04}'/'complete.json')
        if c['steps']!=step or c['cursor']!=step//43*169 or c['epochs_completed']!=step//43 or c['source_sft_updates']!=255 or c['total_sft_updates']!=255 or c['total_dpo_updates']!=11+step:raise ValueError('epoch checkpoint ancestry differs')
    summary,calendar,evidence=audit_results(output,cfg,tok)
    write(output/'evaluation_summary.json',dict(summary,status='awaiting_local_review',**evidence));write(output/'calendar_sql_validation.json',calendar)
    with (output/'results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['branch','mode','kind','episodes','successes'])
        for branch,modes in summary['branches'].items():
            for mode,s in modes.items():
                for kind in ('complete','recovery'):writer.writerow([branch,mode,kind,s[kind]['episodes'],s[kind]['successes']])
    verify_inputs(output);write(output/'audit.json',dict(passed=True,optimizer_updates=cfg['max_steps'],source_sft_updates=255,total_dpo_updates=269,reference_inputs_verified=169,**evidence))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
