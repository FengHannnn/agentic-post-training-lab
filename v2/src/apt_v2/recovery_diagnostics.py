"""Replay-based attribution of recovery actions on fixed train/development contexts."""
import json,hashlib
from collections import Counter
from .environment import Environment,METRICS
from .dpo import action_from_text,failure_reason,TEMPORARY_ERROR
from .dpo_recovery import SCENARIOS,build_context

def messages_id(messages):
    return hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def diagnostic_contexts(tasks):
    return [build_context(t,s,allowed_split=t.split) for t in tasks for s in SCENARIOS]

def diagnose_action(task,context,text,*,context_builder=None,allowed_splits=('train','development')):
    builder=context_builder or (lambda t,s:build_context(t,s,allowed_split=t.split))
    if task.split not in allowed_splits or context!=builder(task,context['scenario']):
        raise ValueError('fixed recovery context in explicitly allowed split required')
    env=Environment(task)
    try:
        for item in context['records']:
            if env.step(action_from_text(item['text']))!=item['row']:raise ValueError('diagnostic prefix replay differs')
        action=action_from_text(text);expert=action_from_text(context['chosen'])
        tool=action.get('tool');args=action.get('arguments')
        tool_label=tool if isinstance(tool,str) else '<invalid>'
        args=args if isinstance(args,dict) else {}
        available={m:sorted(eid for eid,values in env.evidence.items() if m in values) for m in METRICS}
        slots=[]
        if tool=='calculate_change':
            refs=[('current_net_cents',args.get('current_evidence')),('prior_net_cents',args.get('prior_evidence'))]
        elif tool in ('check_result','submit'):
            evidence=args.get('evidence');evidence=evidence if isinstance(evidence,dict) else {}
            refs=[(m,evidence.get(m)) for m in METRICS]
        else:refs=[]
        for metric,eid in refs:
            if eid is None:status='missing_reference'
            elif not isinstance(eid,str):status='invalid_reference_type'
            elif eid not in env.evidence:status='unknown_id'
            elif metric in env.evidence[eid]:status='correct_verified_metric'
            elif not env.evidence[eid]:status='id_exists_unverified'
            else:status='id_exists_wrong_metric'
            slots.append({'metric':metric,'reference':eid,'status':status,'verified_ids_for_metric':available[metric]})
        before=(env.policy_read,env.schema_read,len(env.verified))
        prior=env.records[-1]['action'];verified_before=sorted(env.verified)
        actual=env.step(action);after=(env.policy_read,env.schema_read,len(env.verified))
        reason=failure_reason(actual)
        if reason is None and before==after and actual['observation'].get('passed') is not True:reason='no_progress'
        if actual['tool_error']==TEMPORARY_ERROR:reason=None
        goal=context['chosen_observation'].get('verified_metrics',[]) if expert['tool']=='query_sql' else []
        verified=actual['observation'].get('verified_metrics',[]) if tool=='query_sql' else []
        sql_status=None
        if tool=='query_sql':
            if actual['tool_error']:sql_status='execution_error'
            elif not verified:sql_status='executed_unverified'
            elif goal and set(goal).issubset(verified):sql_status='verified_target'
            elif goal:sql_status='verified_other_metric'
            else:sql_status='verified_any'
        diagnostic={
            'strict_format':set(action)=={'tool','arguments'},'expected_tool':expert['tool'],'actual_tool':tool_label,
            'tool_matches_expert':tool==expert['tool'],'available_verified_metrics':verified_before,
            'available_evidence_ids_by_metric':available,
            'evidence':{'applicable':bool(refs),'slots':slots,
                'all_references_verified':all(s['status']=='correct_verified_metric' for s in slots) if refs else None,
                'all_required_metrics_available':all(available[m] for m,_ in refs) if refs else None},
            'sql':{'attempted':tool=='query_sql','status':sql_status,'goal_metrics':goal,'verified_metrics':verified,
                'target_verified':bool(goal) and set(goal).issubset(verified) if tool=='query_sql' and goal else None},
            'productive':reason is None,'reason':reason,'tool_error':actual['tool_error'],
            'repeated_previous_action':action==prior,
            'failed_checks':[k for k,v in actual['observation'].get('checks',{}).items() if not v]
        }
        return actual,diagnostic
    finally:env.close()

def summarize(rows):
    ds=[r['diagnostic'] for r in rows];evidence=[d for d in ds if d['evidence']['applicable']]
    matched=[d for d in evidence if d['tool_matches_expert']]
    slots=[s for d in evidence for s in d['evidence']['slots']]
    sql=[d for d in ds if d['sql']['attempted']];needed=[d for d in ds if d['expected_tool']=='query_sql']
    return {
        'actions':len(ds),'productive':sum(d['productive'] for d in ds),'strict_format':sum(d['strict_format'] for d in ds),
        'tool_matches_expert':sum(d['tool_matches_expert'] for d in ds),
        'expected_tool_counts':dict(Counter(d['expected_tool'] for d in ds)),
        'actual_tool_counts':dict(Counter(d['actual_tool'] for d in ds)),
        'repeated_previous_action':sum(d['repeated_previous_action'] for d in ds),
        'tool_errors':dict(Counter(d['tool_error'] for d in ds if d['tool_error'])),
        'evidence':{'actions':len(evidence),'all_references_verified':sum(d['evidence']['all_references_verified'] for d in evidence),
            'slots':len(slots),'slot_status_counts':dict(Counter(s['status'] for s in slots)),
            'actions_with_all_required_metrics_available':sum(d['evidence']['all_required_metrics_available'] for d in evidence),
            'expected_tool_actions':len(matched),'expected_tool_all_references_verified':sum(d['evidence']['all_references_verified'] for d in matched)},
        'sql':{'attempts':len(sql),'status_counts':dict(Counter(d['sql']['status'] for d in sql)),
            'any_metric_verified':sum(bool(d['sql']['verified_metrics']) for d in sql),
            'query_required_actions':len(needed),'query_chosen_when_required':sum(d['sql']['attempted'] for d in needed),
            'target_verified_when_required':sum(d['sql']['target_verified'] is True for d in needed)}
    }

def diagnostic_summary(rows,branches):
    return {b:{split:{mode:{'overall':summarize([r for r in rows if r['model_branch']==b and r['split']==split and r['mode']==mode]),
        'by_scenario':{s:summarize([r for r in rows if r['model_branch']==b and r['split']==split and r['mode']==mode and r['scenario']==s]) for s in SCENARIOS}}
        for mode in ('greedy','sample')} for split in ('train','development')} for b in branches}

def comparison_rates(summary):
    def rates(s):
        ratio=lambda n,d:n/d if d else None
        e=s['evidence'];q=s['sql']
        return {'productive_rate':ratio(s['productive'],s['actions']),'expert_tool_rate':ratio(s['tool_matches_expert'],s['actions']),
            'evidence_all_valid_given_expert_tool':ratio(e['expected_tool_all_references_verified'],e['expected_tool_actions']),
            'evidence_slot_valid_rate':ratio(e['slot_status_counts'].get('correct_verified_metric',0),e['slots']),
            'query_chosen_given_required':ratio(q['query_chosen_when_required'],q['query_required_actions']),
            'target_verified_given_required':ratio(q['target_verified_when_required'],q['query_required_actions']),
            'target_verified_given_query_when_required':ratio(q['target_verified_when_required'],q['query_chosen_when_required'])}
    result={}
    for branch,splits in summary.items():
        result[branch]={}
        for mode in ('greedy','sample'):
            result[branch][mode]={}
            for scenario in ('overall',*SCENARIOS):
                get=lambda split:splits[split][mode]['overall'] if scenario=='overall' else splits[split][mode]['by_scenario'][scenario]
                train=rates(get('train'));dev=rates(get('development'))
                result[branch][mode][scenario]={'train':train,'development':dev,
                    'train_minus_development':{k:train[k]-dev[k] if train[k] is not None and dev[k] is not None else None for k in train}}
    return result
