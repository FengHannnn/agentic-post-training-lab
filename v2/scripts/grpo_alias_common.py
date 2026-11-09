"""Matched curriculum identities and measured paired-cycle budget reservation."""
import math
from grpo_common import ROOT,read,validate_config
from grpo_c3_common import states_at,verify_objective_metrics
from grpo_curriculum_common import resume_spent
from grpo_fine_common import cycle_seconds
from apt_v2.grpo_alias_curriculum import BRANCHES,choose_candidate


def matched_configs(config):
    cfg=read(config);branches={b:read(ROOT/p) for b,p in cfg['branches'].items()}
    if (list(branches)!=list(BRANCHES) or cfg['study_phase']!='alias_curriculum'
            or cfg['max_wall_seconds']!=21600 or cfg['groups_per_branch']!=48
            or cfg['packaging_reserve_seconds']!=180 or cfg['midpoint_group_cursor']!=24
            or cfg['selection_rule']!='primary_gain_greedy_guard_then_recovery_v1'
            or cfg['budget_policy']!='paired_cycle_with_midpoint_and_final_reservation_v1'
            or cfg['extra_sft_updates']!=0 or cfg['extra_dpo_updates']!=0):
        raise ValueError('authorized alias comparison budget differs')
    for b,c in branches.items():
        validate_config(c)
        if c['curriculum_arm']!=b or c['run_id']!=cfg['run_id']+'-'+b:raise ValueError('matched alias branch differs')
    ignore={'run_id','curriculum_arm'}
    if {k:v for k,v in branches['control'].items() if k not in ignore}!={k:v for k,v in branches['curriculum'].items() if k not in ignore}:
        raise ValueError('branches differ beyond the target curriculum')
    return cfg,branches


def reserve_seconds(baseline_seconds,midpoint_pending):
    return 2*max(2400.,1.2*baseline_seconds)+240.+(600. if midpoint_pending else 0.)+180.+180.


def training_decision(elapsed,baseline_seconds,stages,midpoint_pending=True):
    reserve=reserve_seconds(baseline_seconds,midpoint_pending);cycle=cycle_seconds(stages)
    return {'continue':21600.-elapsed>=reserve+cycle,'elapsed_seconds':elapsed,
            'estimated_next_paired_cycle_seconds':cycle,'reserved_final_seconds':reserve,
            'midpoint_pending':midpoint_pending,'budget_seconds':21600.,
            'policy':'paired_cycle_with_midpoint_and_final_reservation_v1'}


def verify_training_stop(stopped,cfg,baseline_seconds):
    count=stopped['completed_groups']
    if (stopped['maximum_groups']!=48 or type(count) is not int or not 2<=count<=48 or count%2
            or stopped['policy']!=cfg['budget_policy'] or stopped['reason']!=('group_cap' if count==48 else 'budget_reserve')):
        raise ValueError('frozen equal paired training stop differs')
    if count<48:
        d=stopped['budget_decision']
        if (d is None or d['continue'] is not False or d['policy']!=cfg['budget_policy'] or d['budget_seconds']!=21600.
                or type(d['midpoint_pending']) is not bool or d['midpoint_pending']!=(count<24)
                or not all(math.isfinite(d[k]) for k in ('elapsed_seconds','estimated_next_paired_cycle_seconds','reserved_final_seconds'))
                or not 0<=d['elapsed_seconds']<=21600 or d['estimated_next_paired_cycle_seconds']<600
                or d['reserved_final_seconds']!=reserve_seconds(baseline_seconds,d['midpoint_pending'])
                or 21600.-d['elapsed_seconds']>=d['reserved_final_seconds']+d['estimated_next_paired_cycle_seconds']):
            raise ValueError('budget stop lacks consistent measured reservation')
    return count
