"""Matched identities and conservative paired-cycle budget reservation."""
import math
from grpo_common import ROOT,read,validate_config
from grpo_c3_common import states_at,verify_objective_metrics
from grpo_curriculum_common import resume_spent
from apt_v2.grpo_fine_study import BRANCHES,METHODS,choose_candidate


def matched_configs(config):
    cfg=read(config);branches={b:read(ROOT/p) for b,p in cfg['branches'].items()}
    if (set(branches)!=set(BRANCHES) or cfg['study_phase']!='fine_credit_ablation'
            or cfg['max_wall_seconds']!=21600 or cfg['groups_per_branch']!=48
            or cfg['packaging_reserve_seconds']!=180 or cfg['common_update_probe']!=12
            or cfg['selection_rule']!='primary_gain_greedy_guard_then_recovery_v1'
            or cfg['budget_policy']!='paired_cycle_with_final_reservation_v1'):
        raise ValueError('authorized fine comparison budget differs')
    for b,c in branches.items():
        validate_config(c)
        if c['credit_method']!=METHODS[b] or c['run_id']!=cfg['run_id']+'-'+b or c['max_groups']!=48:
            raise ValueError('matched fine branch identity differs')
    ignore={'run_id','credit_method'}
    if {k:v for k,v in branches['legacy'].items() if k not in ignore}!={k:v for k,v in branches['local'].items() if k not in ignore}:
        raise ValueError('fine branches differ beyond action credit')
    return cfg,branches


def reserve_seconds(baseline_seconds,probe_branches=2):
    # Final two full evaluations, two refreshes, optional small probes, audit and packaging.
    return 2*max(2400.,1.2*baseline_seconds)+240.+300.*probe_branches+180.+180.


def cycle_seconds(stages):
    groups={}
    for row in stages:
        parts=row['stage'].split('-')
        if parts[0] in ('collect','update') and row['exit_code']==0:
            key='-'.join(parts[1:]);groups.setdefault(key,{})[parts[0]]=row['elapsed_seconds']
    complete=[sum(v.values()) for v in groups.values() if set(v)=={'collect','update'}][-8:]
    return 4*max(150.,1.2*sum(complete)/len(complete)) if complete else 720.


def training_decision(elapsed,baseline_seconds,stages):
    reserve=reserve_seconds(baseline_seconds);cycle=cycle_seconds(stages)
    return {'continue':21600.-elapsed>=reserve+cycle,'elapsed_seconds':elapsed,
            'estimated_next_paired_cycle_seconds':cycle,'reserved_final_seconds':reserve,
            'budget_seconds':21600.,'policy':'paired_cycle_with_final_reservation_v1'}


def verify_training_stop(stopped,cfg,baseline_seconds):
    count=stopped['completed_groups']
    if (stopped['maximum_groups']!=48 or not isinstance(count,int) or not 2<=count<=48 or count%2
            or stopped['policy']!=cfg['budget_policy'] or stopped['reason']!=('group_cap' if count==48 else 'budget_reserve')):
        raise ValueError('frozen equal paired training stop differs')
    if count<48:
        d=stopped['budget_decision']
        if (d is None or d['continue'] is not False or d['policy']!=cfg['budget_policy'] or d['budget_seconds']!=21600.
                or not all(math.isfinite(d[k]) for k in ('elapsed_seconds','estimated_next_paired_cycle_seconds','reserved_final_seconds'))
                or not 0<=d['elapsed_seconds']<=21600 or d['estimated_next_paired_cycle_seconds']<600
                or d['reserved_final_seconds']!=reserve_seconds(baseline_seconds)
                or 21600.-d['elapsed_seconds']>=d['reserved_final_seconds']+d['estimated_next_paired_cycle_seconds']):
            raise ValueError('budget stop lacks consistent predeclared reservation decision')
    return count
