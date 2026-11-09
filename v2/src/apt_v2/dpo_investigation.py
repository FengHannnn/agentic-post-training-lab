"""Deterministic probes and endpoint gates for DPO regression diagnosis."""
import json,math,hashlib
from collections import Counter
from .tasks import make_tasks
from .environment import Environment
from .runtime import SYSTEM
from .dpo import corrective_action,TEMPORARY_ERROR
from .dpo_recovery import SCENARIOS,build_context

def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def investigation_tasks():return [t for t in make_tasks() if t.split=='development']
def investigation_contexts():return [build_context(t,s,allowed_split='development') for t in investigation_tasks() for s in SCENARIOS]

def build_probes():
    probes=[]
    for task in [t for t in make_tasks() if t.split in ('train','development')]:
        for scenario in SCENARIOS:
            c=build_context(task,scenario,allowed_split=task.split)
            probes.append({'probe_id':task.id+':'+scenario+':first','kind':'recovery_first','task_id':task.id,'family_id':task.family_id,'split':task.split,'scenario':scenario,
                'expected_tool':json.loads(c['chosen'])['tool'],'messages':c['messages'],'response':c['chosen']})
    for task in investigation_tasks():
        starts=[None]+[build_context(task,s,allowed_split='development') for s in SCENARIOS]
        for context in starts:
            env=Environment(task);messages=[{'role':'system','content':SYSTEM},{'role':'user','content':task.request()}] if context is None else [dict(m) for m in context['messages']]
            try:
                if context is not None:
                    for r in context['records']:
                        from .dpo import action_from_text
                        if env.step(action_from_text(r['text']))!=r['row']:raise ValueError('probe prefix differs')
                index=0
                while not env.done:
                    action=corrective_action(env);text=json.dumps(action);row=env.step(action)
                    if row['tool_error'] not in (None,TEMPORARY_ERROR):raise ValueError('expert probe action failed')
                    scenario='full_task' if context is None else context['scenario'];kind='workflow' if context is None else 'recovery_continuation'
                    probes.append({'probe_id':task.id+':'+scenario+':'+str(index),'kind':kind,'task_id':task.id,'family_id':task.family_id,'split':task.split,'scenario':scenario,
                        'expected_tool':action['tool'],'messages':[dict(m) for m in messages],'response':text})
                    messages.extend([{'role':'assistant','content':text},{'role':'user','content':json.dumps({'observation':row['observation']})}]);index+=1
                if not env.success:raise ValueError('expert probe continuation must finish')
            finally:env.close()
    if len({p['probe_id'] for p in probes})!=len(probes):raise ValueError('distinct probe IDs required')
    return probes

def summarize_pair_metrics(pairs,encoded,metrics,reference,weights):
    if not pairs or len({len(pairs),len(encoded),len(metrics),len(reference),len(weights)})!=1:raise ValueError('aligned scores required')
    def group(indices):
        if not indices:raise ValueError('nonempty score group required')
        rows=[metrics[i] for i in indices];ws=[weights[i] for i in indices]
        result={'pairs':len(rows),'mean_loss':sum(r['loss'] for r in rows)/len(rows),'weighted_mean_loss':sum(w*r['loss'] for w,r in zip(ws,rows))/sum(ws),
            'chosen_ce':-sum(r['chosen_mean_logp'] for r in rows)/len(rows),'rejected_ce':-sum(r['rejected_mean_logp'] for r in rows)/len(rows),'relative_margin':sum(r['relative_margin'] for r in rows)/len(rows)}
        if any(not math.isfinite(v) for v in result.values()):raise ValueError('finite diagnostic metrics required')
        for r in rows:
            z=-.1*r['relative_margin'];loss=max(0,z)+math.log1p(math.exp(-abs(z)))
            if abs(r['loss']-loss)>1e-4:raise ValueError('fixed beta .1 DPO objective differs')
        return result
    return {'overall':group(list(range(len(pairs)))),'by_scenario':{s:group([i for i,p in enumerate(pairs) if p.get('scenario','original_sql')==s]) for s in dict.fromkeys(p.get('scenario','original_sql') for p in pairs)}}

def summarize_probe_metrics(probes,metrics):
    if not probes or len(probes)!=len(metrics):raise ValueError('aligned nonempty probes required')
    if len({p['probe_id'] for p in probes})!=len(probes):raise ValueError('distinct diagnostic probes required')
    for p,m in zip(probes,metrics):
        if m['probe_id']!=p['probe_id'] or m['probe_sha256']!=digest(p) or not m['token_log_probs'] or any(not math.isfinite(v) or v>1e-6 for v in m['token_log_probs']):raise ValueError('probe token binding differs')
        if abs(m['ce']+sum(m['token_log_probs'])/len(m['token_log_probs']))>1e-6:raise ValueError('probe CE differs')
    result={}
    for split in dict.fromkeys(p['split'] for p in probes):
        result[split]={}
        for kind in dict.fromkeys(p['kind'] for p in probes if p['split']==split):
            result[split][kind]={}
            for tool in dict.fromkeys(p['expected_tool'] for p in probes if p['split']==split and p['kind']==kind):
                rows=[m for p,m in zip(probes,metrics) if p['split']==split and p['kind']==kind and p['expected_tool']==tool]
                result[split][kind][tool]={'examples':len(rows),'mean_ce':sum(r['ce'] for r in rows)/len(rows)}
    return result

def episode_summary(episodes):
    rows=[r for e in episodes for r in e['turns']]
    return {'episodes':len(episodes),'successes':sum(e['success'] for e in episodes),'turns':len(rows),'response_tokens':sum(len(r['generation']['response_token_ids']) for r in rows),
        'tools':dict(Counter(r['action'].get('tool','<invalid>') for r in rows)),'tool_errors':sum(e['tool_errors'] for e in episodes)}

def paired(a,b,key):
    x={key(e):e['success'] for e in a};y={key(e):e['success'] for e in b}
    if len(x)!=len(a) or len(y)!=len(b) or x.keys()!=y.keys():raise ValueError('distinct paired keys required')
    return {'both_success':sum(x[k] and y[k] for k in x),'both_failure':sum(not x[k] and not y[k] for k in x),'improved':sum(not x[k] and y[k] for k in x),'regressed':sum(x[k] and not y[k] for k in x)}

def evaluation_summary(complete,recovery,branches):
    result={'branches':{},'paired':{}}
    for branch in branches:
        es=[e for e in complete if e['model_branch']==branch];rs=[e for e in recovery if e['model_branch']==branch]
        result['branches'][branch]={'complete':{mode:episode_summary([e for e in es if e['mode']==mode]) for mode in ('greedy','sample')},
            'recovery_greedy':episode_summary(rs),'recovery_by_scenario':{s:episode_summary([e for e in rs if e['scenario']==s]) for s in SCENARIOS}}
    for baseline in ('parent','weighted_sft'):
        result['paired'][baseline]={}
        for branch in branches:
            if branch in ('parent','weighted_sft'):continue
            pick=lambda rows,b,mode=None:[e for e in rows if e['model_branch']==b and (mode is None or e['mode']==mode)]
            result['paired'][baseline][branch]={'complete':{m:paired(pick(complete,baseline,m),pick(complete,branch,m),lambda e:(e['task_id'],e['rollout_index'])) for m in ('greedy','sample')},
                'recovery_greedy':paired(pick(recovery,baseline),pick(recovery,branch),lambda e:(e['context_id'],e['sample_index']))}
    return result


def metrics_from_scores(current,reference,beta=.1):
    if current['input_id']!=reference['input_id'] or beta!=.1:raise ValueError('fixed reference score binding required')
    for name in ('chosen','rejected'):
        a=current[name];b=reference[name]
        if not a or len(a)!=len(b) or any(not math.isfinite(v) or v>1e-6 for v in a+b):raise ValueError('finite aligned token scores required')
    chosen=sum(current['chosen']);rejected=sum(current['rejected']);margin=(chosen-rejected)-(sum(reference['chosen'])-sum(reference['rejected']))
    z=-beta*margin;loss=max(0,z)+math.log1p(math.exp(-abs(z)))
    return {'loss':loss,'relative_margin':margin,'chosen_mean_logp':chosen/len(current['chosen']),'rejected_mean_logp':rejected/len(current['rejected'])}
