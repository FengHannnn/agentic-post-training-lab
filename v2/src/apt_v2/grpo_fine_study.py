"""Frozen credit-only fine GRPO comparison; fresh development instances."""
from dataclasses import asdict
from .tasks import Task
from .grpo_credit import FINE,LOCAL

SCHEMA='fine_credit_ablation_v1'
BRANCHES=('legacy','local')
METHODS={'legacy':FINE,'local':LOCAL}


def development_tasks():
    from .grpo_curriculum import month_pair
    result=[]
    for i,month in enumerate((2,6,10,1)):
        current,prior=month_pair(2030+i%2,month)
        result.append(Task(f'fine-development-{i:03}','development',f'fine-family-{73000+i}',
            73000+i,current,prior,('West','Central','East','South')[i],
            f'FineDevSegment-{73000+i}','transient_query' if i%2 else 'none'))
    return result


def validate_config(cfg):
    from .grpo_curriculum import SCENARIOS,assert_isolation
    if (cfg.get('curriculum_schema')!=SCHEMA or cfg['study_phase']!='fine_credit_ablation'
            or cfg['algorithm']!='fine_grained_grpo' or cfg['credit_method'] not in METHODS.values()
            or cfg['learning_rate']!=1e-6 or cfg['max_groups']!=48 or cfg['max_updates']!=48
            or cfg['max_wall_seconds']!=21600 or cfg['formal_gpu_budget_seconds']!=21600
            or cfg['group_evaluation_cursors']!=[48] or cfg['packaging_reserve_seconds']!=180
            or cfg['k']!=4 or cfg['recovery_scenarios']!=list(SCENARIOS)
            or cfg.get('closing_evaluation')!={'enabled':False}
            or cfg.get('targeted_evaluation')!={'enabled':True,'seed':2026110819,'samples_per_context':1}):
        raise ValueError('frozen credit-only fine study differs')
    assert_isolation(cfg)


def method_record(cfg):
    return {'method':cfg['credit_method'],'study':SCHEMA,'learning_rate':1e-6,
        'formula':('normalized episode outcome plus 0.2 per newly verified metric at this action'
                   if cfg['credit_method']==LOCAL else 'terminal-zero potential RTG normalized at relative turn'),
        'new_method_process_scale':.2,'extra_sft_updates':0,'extra_dpo_updates':0,
        'new_method_is_potential_equivalent':False,'comparison':'same sampled group count; actual updates reported separately',
        'common_update_probe':12,'probe_used_for_selection':False}


def probe_plan(cfg):
    from .grpo_evaluation import evaluation_plan
    from .grpo_curriculum import targeted_plan
    complete=[x for x in evaluation_plan(cfg) if x['scenario'] is None and x['mode']=='sample' and x['rollout_index']==0]
    target=[x for x in targeted_plan(cfg) if x['mode']=='sample']
    return [{**x,'seed':x['seed']+80000000} for x in complete+target]


def choose_candidate(summaries):
    base=summaries['weighted_sft'];primary='recovery' if base['sample']['complete']['successes']==base['sample']['complete']['episodes'] else 'complete'
    def score(b,m,k):return summaries[b][m][k]['successes']
    checks={b:{'greedy_nonregression':all(score(b,'greedy',k)>=score('weighted_sft','greedy',k) for k in ('complete','recovery')),
               'sample_nonregression':all(score(b,'sample',k)>=score('weighted_sft','sample',k) for k in ('complete','recovery')),
               'primary_gain':score(b,'sample',primary)-score('weighted_sft','sample',primary)} for b in BRANCHES}
    eligible=[b for b in BRANCHES if checks[b]['greedy_nonregression'] and checks[b]['sample_nonregression'] and checks[b]['primary_gain']>0]
    ranked=sorted(eligible,key=lambda b:(checks[b]['primary_gain'],score(b,'sample','recovery'),b=='legacy'),reverse=True)
    return {'scope':'development_selection','primary_metric':primary,'checks':checks,
            'selected_candidate':ranked[0] if ranked else None,
            'local_minus_legacy_primary':score('local','sample',primary)-score('legacy','sample',primary),
            'targeted_or_probe_used_for_selection':False,'future_confirmation':'new frozen tasks before results'}
