"""Turn returns and leave-one-out episode baselines for on-policy agent updates."""
import math

def discounted_returns(records,gamma=1.):
    if not records or not records[-1]['done'] or any(r['done'] for r in records[:-1]):
        raise ValueError('one complete episode is required')
    if not 0<gamma<=1:raise ValueError('gamma must be in (0,1]')
    values=[];value=0.
    for row in reversed(records):
        reward=row['reward']
        if not math.isfinite(reward):raise ValueError('finite rewards required')
        value=reward+gamma*value;values.append(value)
    return list(reversed(values))

def assign_turn_credit(episodes):
    if len(episodes)<2 or len({e['task_id'] for e in episodes})!=1:
        raise ValueError('at least two independent rollouts of the same task are required')
    totals=[sum(r['outcome_reward'] for r in e['turns']) for e in episodes]
    result=[]
    for index,episode in enumerate(episodes):
        baseline=(sum(totals)-totals[index])/(len(totals)-1)
        returns=discounted_returns(episode['turns'])
        result.append({'task_id':episode['task_id'],'baseline':baseline,'turns':[
            {'turn':row['turn'],'return':value,'advantage':value-baseline}
            for row,value in zip(episode['turns'],returns)]})
    return result

def training_samples(episode):
    """Only stochastic policy actions with aligned native probabilities enter training."""
    result=[]
    for row in episode['turns']:
        sample=row.get('generation',{})
        ids=sample.get('response_token_ids',[])
        probs=sample.get('behavior_token_log_probs')
        if sample.get('mode')!='sample' or not ids or probs is None or len(ids)!=len(probs):
            raise ValueError('stochastic native action tokens and behavior probabilities required')
        prompt=sample.get('prompt_token_ids')
        if not prompt or any(type(v) is not int or v<0 for v in prompt+ids):
            raise ValueError('native prompt and response token IDs required')
        if any(not math.isfinite(v) or v>1e-6 for v in probs):
            raise ValueError('finite behavior log probabilities required')
        result.append({'turn':row['turn'],'prompt_token_ids':sample['prompt_token_ids'],
            'response_token_ids':ids,'response_mask':[True]*len(ids),
            'behavior_token_log_probs':probs})
    return result
