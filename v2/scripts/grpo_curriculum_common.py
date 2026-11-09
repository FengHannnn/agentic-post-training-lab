"""Matched learning-rate study, cumulative budget and frozen development selection."""
import time
from grpo_common import ROOT,read,validate_config
from grpo_c3_common import states_at,verify_objective_metrics

BRANCHES=('control','lr10x')


def matched_configs(config):
    cfg=read(config)
    branches={b:read(ROOT/p) for b,p in cfg['branches'].items()}
    if (set(branches)!=set(BRANCHES) or cfg['study_phase']!='outcome_curriculum'
            or cfg['max_wall_seconds'] not in (14400,21600,25200)
            or cfg['groups_per_branch']!=(24 if cfg['max_wall_seconds']==14400 else 48)
            or cfg['packaging_reserve_seconds']<180
            or cfg.get('selection_rule')!='primary_gain_greedy_guard_then_recovery_v1'):
        raise ValueError('authorized matched curriculum job differs')
    for b,c in branches.items():
        validate_config(c)
        if (c['study_phase']!=cfg['study_phase'] or c['max_groups']!=cfg['groups_per_branch']
                or c['max_wall_seconds']!=cfg['max_wall_seconds'] or c['required_gpu_name']!='NVIDIA L40S'
                or c['learning_rate']!={'control':1e-6,'lr10x':1e-5}[b]
                or c['run_id']!=cfg['run_id']+'-'+b):
            raise ValueError('matched curriculum branch differs')
    ignore={'run_id','learning_rate'}
    if {k:v for k,v in branches['control'].items() if k not in ignore}!={k:v for k,v in branches['lr10x'].items() if k not in ignore}:
        raise ValueError('curriculum branches must differ only in learning rate')
    return cfg,branches


def resume_spent(prior,status=None,now=None):
    spent=max(prior.get('elapsed_seconds',0),(status or {}).get('elapsed_seconds',0))
    active=prior.get('active_started_unix')
    if active is not None:spent+=max(0,(time.time() if now is None else now)-active)
    return spent


def select_candidate(summaries):
    baseline=summaries['weighted_sft']
    full=baseline['sample']['complete']
    primary='recovery' if full['successes']==full['episodes'] else 'complete'
    def score(label,mode,kind):return summaries[label][mode][kind]['successes']
    guards={b:all(score(b,'greedy',kind)>=score('weighted_sft','greedy',kind) for kind in ('complete','recovery')) for b in BRANCHES}
    gains={b:score(b,'sample',primary)-score('weighted_sft','sample',primary) for b in BRANCHES}
    sample_guards={b:all(score(b,'sample',kind)>=score('weighted_sft','sample',kind) for kind in ('complete','recovery')) for b in BRANCHES}
    eligible=[b for b in BRANCHES if guards[b] and sample_guards[b] and gains[b]>0]
    # Standard evaluation alone selects a candidate; targeted recovery remains diagnostic.
    ranked=sorted(eligible,key=lambda b:(gains[b],score(b,'sample','recovery'),score(b,'sample','complete'),b=='control'),reverse=True)
    return {'primary_metric':primary,'greedy_guards':guards,'sample_nonregression_guards':sample_guards,'primary_gain_successes':gains,
            'selected_candidate':ranked[0] if ranked else None,'scope':'development_selection',
            'next_confirmation':'fresh_frozen_tasks_before_results'}



def configuration_profile(hours):
    if hours not in (4,6,7):raise ValueError('authorized budget profile must be 4, 6 or 7 hours')
    import copy
    groups=24 if hours==4 else 48
    run_id='grpo-outcome-curriculum-v1-20261107'
    base=read(ROOT/'configs/grpo-c3-outcome-v1.json')
    from apt_v2.grpo_curriculum import SCHEMA,SCENARIOS
    report='reports/grpo-outcome-curriculum-contract-v1-20261107.md'
    validation='reports/grpo-outcome-curriculum-validation-v1-20261107.json'
    base.update(study_phase='outcome_curriculum',curriculum_schema=SCHEMA,seed=2026110717,
        max_wall_seconds=hours*3600,formal_gpu_budget_seconds=hours*3600,packaging_reserve_seconds=180,
        max_groups=groups,max_updates=groups,group_evaluation_cursors=[groups],recovery_scenarios=list(SCENARIOS),
        contract_report=report,validation_report=validation,closing_evaluation={'enabled':False},
        targeted_evaluation={'enabled':True,'seed':2026110719,'samples_per_context':1})
    base['eval']['seed']=2026110710
    base['sample_eval']['seed']=2026110711
    base['recovery_eval']['seed']=2026110712
    base['evaluation']['seed']=2026110711
    branches={}
    for b in BRANCHES:
        c=copy.deepcopy(base);c.update(run_id=run_id+'-'+b,learning_rate={'control':1e-6,'lr10x':1e-5}[b])
        validate_config(c);branches[b]=c
    cfg={'run_id':run_id,'study_phase':'outcome_curriculum','max_wall_seconds':hours*3600,
         'packaging_reserve_seconds':180,'groups_per_branch':groups,
         'branches':{b:'configs/grpo-outcome-curriculum-'+b+'-v1.json' for b in BRANCHES},
         'contract_report':report,'validation_report':validation,
         'selection_rule':'primary_gain_greedy_guard_then_recovery_v1',
         'clone_policy':'new_L40S_allowed_at_start_same_GPU_software_and_driver_within_job',
         'comparison':'same_expanded_curriculum_and_seeds_learning_rate_only',
         'extra_sft_updates':0,'extra_dpo_updates':0,'future_confirmation':'new_frozen_tasks'}
    return cfg,branches
