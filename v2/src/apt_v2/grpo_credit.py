"""Matched outcome credit and potential-shaped, depth-normalized action credit."""
import math
from .grpo import group_advantages

OUTCOME='outcome'
FINE='potential_rtg_depth_v1'
LOCAL='outcome_local_verified_v2'


def local_credit(episodes,base):
    from .environment import METRICS
    local=[];advantages=[]
    for i,e in enumerate(episodes):
        seen={m for item in e['start']['prefix'] for m in item['row']['new_verified_metrics']}
        rewards=[];values=[]
        for row in e['turns']:
            new=row['new_verified_metrics']
            if (not isinstance(new,list) or len(new)!=len(set(new)) or set(new)-set(METRICS)
                    or seen.intersection(new) or not math.isclose(row['potential_before'],.2*len(seen),abs_tol=1e-12)):
                raise ValueError('unique newly verified metric transition required')
            seen.update(new)
            after=0. if row['done'] else .2*len(seen)
            if (not math.isclose(row['potential_after'],after,abs_tol=1e-12)
                    or not math.isclose(row['process_reward'],after-row['potential_before'],abs_tol=1e-12)):
                raise ValueError('verified progress and terminal potential binding differs')
            reward=.2*len(new)
            rewards.append(reward);values.append(base['advantages'][i]+reward)
        local.append(rewards);advantages.append(values)
    eligible=any(a!=0 for values in advantages for a in values)
    return {**base,'method':LOCAL,'outcome_zero_variance':base['zero_variance'],
            'zero_variance':not eligible,'eligible':eligible,'turn_advantages':advantages,
            'local_progress_rewards':local,'local_progress_scale':.2,
            'eligibility_basis':'nonzero_combined_action_advantage'}

def compute_credit(episodes,k=4,epsilon=1e-8,method=OUTCOME):
    base=group_advantages(episodes,k,epsilon)
    if method==OUTCOME: return base
    if method==LOCAL:return local_credit(episodes,base)
    if method!=FINE: raise ValueError('unknown GRPO credit method')
    returns=[]; shaped=[]
    for e in episodes:
        rewards=[]
        for row in e['turns']:
            before=row['potential_before']; after=row['potential_after']
            if (not math.isfinite(before) or not math.isfinite(after)
                    or not math.isclose(row['process_reward'],after-before,abs_tol=1e-12)):
                raise ValueError('verified potential delta differs')
            rewards.append(row['outcome_reward']+after-before)
        if e['turns'][-1]['potential_after']!=0:
            raise ValueError('terminal potential must reset to zero')
        rtg=[math.fsum(rewards[t:]) for t in range(len(rewards))]
        expected=math.fsum(r['outcome_reward'] for r in e['turns'])-e['turns'][0]['potential_before']
        if not math.isclose(rtg[0],expected,abs_tol=1e-10):
            raise ValueError('potential telescoping identity differs')
        returns.append(rtg); shaped.append(rewards)
    advantages=[[0.]*len(x) for x in returns]; depths=[]
    for t in range(max(map(len,returns))):
        active=[i for i,r in enumerate(returns) if t<len(r)]
        values=[returns[i][t] for i in active]; mean=math.fsum(values)/len(values)
        std=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
        eligible=len(active)>=2 and max(values)!=min(values)
        if eligible:
            for i in active: advantages[i][t]=(returns[i][t]-mean)/(std+epsilon)
        depths.append({'relative_turn':t,'active_episodes':active,'mean':mean,'population_std':std,'eligible':eligible})
    eligible=any(d['eligible'] for d in depths)
    return {**base,'method':method,'outcome_zero_variance':base['zero_variance'],
            'zero_variance':not eligible,'eligible':eligible,'turn_returns':returns,
            'turn_advantages':advantages,'shaped_rewards':shaped,'depth_statistics':depths}


def action_advantage(credit,episode_index,relative_turn):
    if credit.get('method') in (FINE,LOCAL): return credit['turn_advantages'][episode_index][relative_turn]
    return credit['advantages'][episode_index]
