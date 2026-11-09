"""Shared C3 matched-study identity and independent cached objective audit."""
import math
from grpo_common import ROOT,read,verify_files,validate_config,file_hash,initial_policy,load_transition
from apt_v2.grpo_credit import action_advantage


def matched_configs(config):
    cfg=read(config); branches={b:read(ROOT/p) for b,p in cfg['branches'].items()}
    if (set(branches)!={'outcome','fine'} or cfg['max_wall_seconds']>14400
            or cfg['groups_per_branch']!=24 or cfg['packaging_reserve_seconds']<120):
        raise ValueError('authorized one-job matched budget differs')
    for b,c in branches.items():
        validate_config(c)
        if c['study_phase']!='C3' or c['max_groups']!=24 or c['group_evaluation_cursors']!=[24]:raise ValueError('C3 branch schedule differs')
        if c['credit_method']!=('outcome' if b=='outcome' else 'potential_rtg_depth_v1'):raise ValueError('matched branch credit differs')
        if c['required_gpu_name']!='NVIDIA L40S':raise ValueError('new L40S class required')
    ignore={'run_id','algorithm','credit_method'}
    if {k:v for k,v in branches['outcome'].items() if k not in ignore}!={k:v for k,v in branches['fine'].items() if k not in ignore}:
        raise ValueError('matched branches differ beyond action credit')
    return cfg,branches


def states_at(output):
    cfg=read(output/'config.json'); states=[initial_policy(cfg)]
    for i in range(cfg['max_groups']):
        if not (output/'transitions'/f'group-{i:04}.json').exists():break
        states.append(load_transition(output,i,states[-1])['after'])
    return states


def verify_objective_metrics(output):
    import torch
    from apt_v2.grpo_optimizer import action_objective
    cfg=read(output/'config.json'); maximum=0.; verified=0
    for path in sorted((output/'transitions').glob('group-*.json')):
        tr=read(path)
        if not tr['updated']:continue
        group=read(output/'groups'/path.name); credit=group['credit']; cache=read(output/tr['after']['checkpoint']/'scores.json')
        stats={'loss':0.,'sampled_k3':0.,'clip_fraction':0.,'tis_mean':0.,'tis_cap_fraction':0.,'old_minus_q_abs_mean':0.,'old_minus_q_abs_max':0.}
        for entry in cache['entries']:
            i=entry['episode'];old,ref,q=[torch.tensor(entry[k]) for k in ('old','reference','q')]
            relative=entry['turn']-group['episodes'][i]['prefix_turns']
            loss,metrics=action_objective(old,old,ref,q,action_advantage(credit,i,relative),cfg['beta'],cfg['clip'],cfg['tis_cap'])
            weight=1/(cfg['k']*len(group['episodes'][i]['turns']));stats['loss']+=loss.item()*weight
            for key,v in metrics.items():stats[key]=max(stats[key],v) if key.endswith('_max') else stats[key]+v*weight
        error=max(abs(value-tr['metrics'][key]) for key,value in stats.items());maximum=max(maximum,error)
        if error>5e-5:raise ValueError('independent cached objective recomputation differs')
        verified+=1
    return {'passed':True,'actual_updates_recomputed':verified,'maximum_metric_difference':maximum}
