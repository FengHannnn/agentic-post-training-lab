"""Read-only SFT diagnosis: probes, suffix scores and episode summaries."""
import json,math
from collections import defaultdict,Counter
from .runtime import parse_action

def select_probes(examples):
    probes=[]
    # Same frozen training contexts across engines, with one recovery context.
    for tool in ('read_policy','inspect_schema','query_sql','calculate_change','check_result','submit'):
        probes.append(next(e for e in examples if e['variant']=='expert' and e['task_id']=='train-000'
            and parse_action(e['response'])['tool']==tool))
    probes.append(next(e for e in examples if e['task_id']=='train-000' and e['variant']=='sql_repair'
        and 'no such table: metrics' in e['messages'][-1]['content']))
    probes.append(next(e for e in examples if e['task_id']=='train-000' and e['variant']=='evidence_repair'
        and 'provide verified monthly evidence IDs' in e['messages'][-1]['content']))
    return [{'probe_id':f'probe-{i:02}',**e} for i,e in enumerate(probes)]

def score_action(model,sample):
    import torch
    device=next(model.parameters()).device;n=sample['response_length']
    ids=torch.tensor([sample['input_ids']],device=device)
    with torch.inference_mode():
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
            logits_to_keep=n+1).logits[0,-n-1:-1].float()
        target=torch.tensor(sample['labels'][-n:],device=device)
        logp=logits.log_softmax(-1).gather(-1,target[:,None]).squeeze(-1)
        top=logits.argmax(-1)
    return {'token_log_probs':logp.cpu().tolist(),'argmax_token_ids':top.cpu().tolist(),
        'tokens':n,'nll_sum':-logp.sum().item(),'correct_tokens':int((top==target).sum()),
        'all_tokens_correct':bool((top==target).all())}

def grouped_scores(rows):
    groups=defaultdict(list)
    for row in rows:
        for key in ('all','tool:'+row['tool'],'variant:'+row['variant']):groups[key].append(row)
    result={}
    for key,items in groups.items():
        tokens=sum(r['tokens'] for r in items);nll=sum(r['nll_sum'] for r in items)
        result[key]={'actions':len(items),'tokens':tokens,'token_mean_nll':nll/tokens,
            'action_mean_nll':sum(r['nll_sum']/r['tokens'] for r in items)/len(items),
            'token_accuracy':sum(r['correct_tokens'] for r in items)/tokens,
            'teacher_forced_exact_actions':sum(r['all_tokens_correct'] for r in items)}
    return result

def summarize_episodes(episodes):
    rows=[r for e in episodes for r in e['turns']];formatted=0
    for r in rows:
        try:parse_action(r['generation']['text']);formatted+=1
        except (ValueError,TypeError):pass
    return {'tasks':len(episodes),'successes':sum(e['success'] for e in episodes),
        'fault_tasks':sum(e['fault_mode']!='none' for e in episodes),'recovered':sum(e['recovered'] for e in episodes),
        'turns':len(rows),'strict_format':formatted,'tool_errors':sum(e['tool_errors'] for e in episodes),
        'valid_sql':sum(r['action'].get('tool')=='query_sql' and bool(r['observation'].get('verified_metrics')) for r in rows),
        'verified_metric_events':sum(len(r['new_verified_metrics']) for r in rows),
        'response_tokens':sum(len(r['generation'].get('response_token_ids') or []) for r in rows),
        'errors':dict(Counter(r['tool_error'] for r in rows if r['tool_error']))}

def compare_scores(left,right):
    if len({len(left['token_log_probs']),len(right['token_log_probs']),len(left['argmax_token_ids']),len(right['argmax_token_ids'])})!=1:raise ValueError('unaligned probability lengths')
    gaps=[abs(a-b) for a,b in zip(left['token_log_probs'],right['token_log_probs'])]
    if not gaps or any(not math.isfinite(v) for v in gaps):raise ValueError('finite nonempty score comparison required')
    return {'tokens':len(gaps),'max_abs_logp_difference':max(gaps),'mean_abs_logp_difference':sum(gaps)/len(gaps),
        'argmax_disagreements':sum(a!=b for a,b in zip(left['argmax_token_ids'],right['argmax_token_ids']))}


def probe_record(probe_id,mode,sample,score):
    from dataclasses import asdict
    generation=asdict(sample)
    sampling_mode=generation.pop('mode')
    if mode not in ('base','sft'):raise ValueError('base/sft model branch required')
    return {**generation,'probe_id':probe_id,'mode':mode,'sampling_mode':sampling_mode,'score':score}


def index_probes(rows,probe_ids):
    expected={(mode,pid) for mode in ('base','sft') for pid in probe_ids}
    indexed={}
    for row in rows:
        key=(row['mode'],row['probe_id'])
        if key in indexed:raise ValueError('duplicate probe identity: '+str(key))
        if key not in expected:raise ValueError('unexpected probe identity: '+str(key))
        indexed[key]=row
    if set(indexed)!=expected:raise ValueError('missing probe identity')
    return indexed
