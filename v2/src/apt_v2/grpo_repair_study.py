"""Frozen verifier-repair study: diagnostic routing, reminder fading and warm-start gates."""
import copy
from dataclasses import asdict, dataclass
from .tasks import Task
from .sql_verifier import VERSION
from .grpo_credit import LOCAL, compute_credit
from .dpo_recovery import SCENARIOS as STANDARD

SCHEMA = 'repair_study_v1'
W_SHA = 'f8fb95a4135af76398a1ab9c84f49b8f90b7e30ce77cc50a4b59f863202d689f'
COMPOUND = 'repair_compound'
REMINDED = 'repair_reminded'
SIMPLIFIED = 'repair_simplified'
CONDITIONS = (COMPOUND, REMINDED, SIMPLIFIED)
REMINDER = ('Check the returned column aliases and verified_metrics for BOTH monthly queries. '
            'The required aliases are current_net_cents and prior_net_cents. '
            'A successful SQL execution with empty verified_metrics is not monthly evidence. '
            'Repair the queries before calculate_change, and use actual verified evidence IDs.')


@dataclass(frozen=True)
class RepairTask(Task):
    verifier_version: str = VERSION

    def public(self):
        result = super().public()
        result.pop('verifier_version', None)
        return result


def training_tasks(cfg=None):
    from .grpo_alias_curriculum import training_tasks as prior
    return [RepairTask(**asdict(t)) for t in prior({})]


def development_tasks(cfg=None):
    if cfg and cfg.get('repair_profile') == 'fine_closeout_credit_v1':
        from .grpo_fine_closeout import panel_tasks
        return panel_tasks(cfg['evaluation_panel'])
    from .grpo_curriculum import month_pair
    result = []
    for i, month in enumerate((1, 4, 8, 12)):
        current, prior = month_pair(2036+i%2, month)
        result.append(RepairTask(f'repair-development-{i:03}', 'development', f'repair-family-{83000+i}',
                                 83000+i, current, prior, ('Central','East','West','South')[i],
                                 f'RepairDevSegment-{83000+i}', 'transient_query' if i%2 else 'none'))
    return result


def assert_isolation(cfg=None):
    from .tasks import make_tasks
    from .grpo_curriculum import development_tasks as curriculum_dev
    from .grpo_fine_study import development_tasks as fine_dev
    from .grpo_alias_curriculum import development_tasks as alias_dev
    from .grpo_confirmation import confirmation_tasks
    train, dev = training_tasks(), development_tasks(cfg)
    excluded = [t for t in make_tasks() if t.split != 'train'] + curriculum_dev() + fine_dev() + alias_dev()
    excluded += confirmation_tasks({'protocol_version':2, 'task_count':12})
    if cfg and cfg.get('repair_profile') == 'fine_closeout_credit_v1':
        from .grpo_fine_closeout import panel_tasks
        other = 'confirmation' if cfg['evaluation_panel'] == 'development' else 'development'
        excluded += development_tasks() + panel_tasks(other)
    for field in ('id','family_id','seed','segment'):
        a,b,c = ({getattr(t,field) for t in ts} for ts in (train,dev,excluded))
        if len(a)!=12 or len(b)!=4 or a&b or a&c or b&c:
            raise ValueError('repair study split overlap: '+field)
    return {'passed':True,'train_tasks':12,'development_tasks':4,'excluded_instances':len(excluded)}


def context(task, scenario, allowed_split='train'):
    if not isinstance(task, RepairTask) or task.verifier_version != VERSION:
        raise ValueError('explicit repair verifier task required')
    if scenario and scenario.startswith('credit_'):
        from .grpo_fine_closeout import context as credit_context
        return credit_context(task, scenario, allowed_split)
    if scenario not in CONDITIONS:
        from .grpo_curriculum import build_context
        return build_context(task, scenario, allowed_split)
    from .grpo_alias_curriculum import context as alias_context
    original = 'missing_alias_pair__simplified' if scenario == SIMPLIFIED else 'missing_alias_pair__compound'
    result = copy.deepcopy(alias_context(task, original, allowed_split))
    result.update(context_id=task.id+':'+scenario, scenario=scenario)
    if scenario == REMINDED:
        result['messages'].append({'role':'user','content':REMINDER})
    return result


def diagnostic_tasks():
    tasks = training_tasks()
    # The prior pool is interleaved old/new; pick four of each, balanced for transient faults.
    old = [t for t in tasks if t.id.startswith('train-')]
    new = [t for t in tasks if not t.id.startswith('train-')]
    if len(old)<4 or len(new)<4:
        raise ValueError('four old and four new training tasks required')
    return [t for pair in zip(old[:4],new[:4]) for t in pair]


def diagnostic_plan(cfg):
    return [{'index': ci*8+ti,'task_id':t.id,'scenario':scenario,'kind':'recovery',
             'seed':cfg['diagnostic_seed']+ti*100000,'condition':scenario}
            for ci,scenario in enumerate(CONDITIONS) for ti,t in enumerate(diagnostic_tasks())]


def route_diagnostic(groups):
    from .grpo import digest
    if len(groups)!=24:
        raise ValueError('all 24 diagnostic K4 groups required')
    result={}
    expected={(s,t.id) for s in CONDITIONS for t in diagnostic_tasks()}
    if {(g['spec']['scenario'],g['spec']['task_id']) for g in groups} != expected:
        raise ValueError('diagnostic condition/task coverage differs')
    for condition in CONDITIONS:
        rows=[g for g in groups if g['spec']['scenario']==condition]
        first=eligible=0
        for g in rows:
            if len(g['episodes'])!=4:
                raise ValueError('diagnostic group requires K4')
            credit=compute_credit(g['episodes'],4,1e-8,LOCAL)
            if credit != g['credit']:
                raise ValueError('diagnostic credit differs')
            first += sum(bool(set(e['turns'][0]['new_verified_metrics']) &
                              {'current_net_cents','prior_net_cents'}) for e in g['episodes'])
            progress=any(r['new_verified_metrics'] for e in g['episodes'] for r in e['turns'])
            eligible += bool(progress and credit['eligible'])
        result[condition]={'episodes':32,'first_action_certifications':first,
                           'progress_credit_groups':eligible,'qualified':first>=8 and eligible>=4}
    raw,hint=result[COMPOUND],result[REMINDED]
    if raw['qualified']:
        route='online_plain'
    elif hint['qualified'] and hint['first_action_certifications']-raw['first_action_certifications']>=4:
        route='online_reminder'
    else:
        route='warm_start'
    return {'route':route,'conditions':result,'groups_sha256':digest(groups),
            'thresholds':{'first_action':8,'progress_credit_groups':4,'reminder_net_gain':4}}


def start_spec(cfg,index):
    if cfg.get('repair_profile') == 'fine_closeout_credit_v1':
        from .grpo_fine_closeout import start_spec as credit_spec
        return credit_spec(cfg,index)
    if type(index) is not int or not 0<=index<=36:
        raise ValueError('bounded repair group cursor required')
    pair=index//2;phase=min(index//12,2);task=training_tasks()[pair%12]
    target=bool(index%2)
    if target:
        reminded=cfg['repair_arm']=='curriculum' and (phase==0 or (phase==1 and pair%6 in (0,3,4)))
        scenario=REMINDED if reminded else COMPOUND
    else:
        complete=(pair%2==0)==(phase%2==0)
        scenario=None if complete else STANDARD[(pair//2)%len(STANDARD)]
    return {'index':index,'task_id':task.id,'scenario':scenario,'kind':'recovery' if scenario else 'complete',
            'seed':cfg['seed']+index*100000,'phase':phase+1,'training_role':'target' if target else 'maintenance'}


def targeted_plan(cfg):
    return [{'task_id':t.id,'scenario':COMPOUND,'kind':'targeted_recovery','mode':mode,'rollout_index':ri,
             'seed':cfg['targeted_evaluation']['seed']+ti*100000+ri*100,
             'temperature':0 if mode=='greedy' else cfg['recovery_eval']['temperature']}
            for ti,t in enumerate(development_tasks(cfg)) for mode in ('greedy','sample')
            for ri in range(1 if mode=='greedy' else 3)]


def summarize_targeted(episodes):
    def stats(rows):
        return {'episodes':len(rows),'successes':sum(e['success'] for e in rows),
                'first_action_certifications':sum(bool(set(e['turns'][0]['new_verified_metrics']) &
                    {'current_net_cents','prior_net_cents'}) for e in rows),
                'progress_episodes':sum(any(r['new_verified_metrics'] for r in e['turns']) for e in rows)}
    return {mode:stats([e for e in episodes if e['evaluation_mode']==mode]) for mode in ('greedy','sample')}


def learning_rate_protocol(profile):
    if profile == 'lr_v1':
        from . import grpo_lr_study
        return grpo_lr_study
    if profile == 'lr_midpoint_v1':
        from . import grpo_lr_midpoint
        return grpo_lr_midpoint
    if profile == 'closeout_v1':
        from . import grpo_closeout_study
        return grpo_closeout_study
    return None


def warm_recipe_counts(cfg):
    profile=cfg.get('repair_profile')
    if profile=='dose_v1':
        from .grpo_dose_study import recipe
        r=recipe(cfg.get('warm_target_repeats'));return r['steps'],r['steps']*4
    protocol = learning_rate_protocol(profile)
    if protocol is not None:
        r = protocol.recipe(cfg.get('warm_learning_rate'))
        if cfg.get('warm_target_repeats') != r['target_repeats']:
            raise ValueError('learning-rate branch requires the fixed r16 dose')
        return r['steps'], r['steps'] * r['batch_size']
    return (15,60) if profile=='retention_v1' else (24,96)


def expected_source_sft_updates(cfg):
    if cfg.get('curriculum_schema')!=SCHEMA or cfg.get('repair_arm')!='warm':return 513
    return 513+warm_recipe_counts(cfg)[0]


def validate_config(cfg):
    profile=cfg.get('repair_profile')
    if profile == 'fine_closeout_credit_v1':
        from .grpo_fine_closeout import validate_branch
        validate_branch(cfg)
        assert_isolation(cfg)
        return
    protocol = learning_rate_protocol(profile)
    if profile not in (None, 'retention_v1', 'dose_v1') and protocol is None:
        raise ValueError('unknown repair profile')
    warm_updates=513+warm_recipe_counts(cfg)[0]
    if profile=='retention_v1' and not cfg['run_id'].startswith('grpo-sql-retention-study-v1-20261110-'):
        raise ValueError('retention branch run identity differs')
    if profile=='dose_v1' and not cfg['run_id'].startswith('grpo-sql-dose-study-v1-20261110-'):
        raise ValueError('dose branch run identity differs')
    max_wall_seconds = 21600
    if protocol is not None:
        max_wall_seconds = protocol.MAX_WALL_SECONDS
        prefix = protocol.RUN_ID + '-'
        if not cfg['run_id'].startswith(prefix):
            raise ValueError('learning-rate branch run identity differs')
        suffix = cfg['run_id'].removeprefix(prefix)
        if cfg['repair_arm'] == 'warm':
            if protocol.CANDIDATES.get(suffix) != cfg['warm_learning_rate']:
                raise ValueError('learning-rate candidate identity differs')
        elif suffix not in ('w', 'template'):
            raise ValueError('learning-rate W identity differs')
    if (cfg.get('curriculum_schema')!=SCHEMA or cfg.get('verifier_version')!=VERSION
            or cfg.get('repair_arm') not in ('control','curriculum','plain','warm')
            or cfg['credit_method']!=LOCAL or cfg['learning_rate']!=1e-6 or cfg['k']!=4
            or cfg['max_groups']!=36 or cfg['max_updates']!=36 or cfg['group_evaluation_cursors']!=[36]
            or cfg['max_wall_seconds']!=max_wall_seconds or cfg['minimum_actual_updates']!=0
            or cfg['source_sft_updates'] not in (513,warm_updates) or cfg['source_dpo_updates']!=11
            or cfg['source_sft_updates'] != (warm_updates if cfg['repair_arm']=='warm' else 513)
            or cfg['closing_evaluation']!={'enabled':False}):
        raise ValueError('frozen repair branch contract differs')
    if cfg.get('baseline_W_sha256')!=W_SHA or (cfg['repair_arm']!='warm' and cfg['adapter_sha256']!=W_SHA):
        raise ValueError('formal W ancestry differs')
    if cfg['repair_arm']=='warm' and cfg['adapter_sha256']==W_SHA:
        raise ValueError('warm reference must be the actual changed B')
    assert_isolation()


def frozen_inputs(cfg):
    from .grpo import initial_state
    from .dpo import corrective_action
    proofs=[]
    states = ()
    if cfg.get('repair_profile') == 'fine_closeout_credit_v1':
        from .grpo_fine_closeout import STATES
        states = STATES
    for t in training_tasks()+development_tasks(cfg):
        for scenario in (None,*STANDARD,*CONDITIONS,*(states if t.split == 'train' else ())):
            env,_,_,_=initial_state(t,scenario,context_builder=context)
            try:
                prefix=len(env.records)
                while not env.done:env.step(corrective_action(env))
                if not env.success:raise ValueError('repair expert exceeds total budget')
                proofs.append({'task_id':t.id,'scenario':scenario,'prefix_turns':prefix,
                               'continuation_turns':len(env.records)-prefix,'total_turns':len(env.records),'passed':True})
            finally:env.close()
    return {'curriculum_training_tasks.json':[asdict(t) for t in training_tasks()],
            'curriculum_group_plan.json':[start_spec(cfg,i) for i in range(cfg.get('max_groups',36)+1)],
            'curriculum_isolation.json':assert_isolation(cfg),'curriculum_expert_proof.json':proofs,
            'targeted_evaluation_plan.json':targeted_plan(cfg),
            'repair_method.json':{'verifier_version':VERSION,'reminder':REMINDER,'extra_sft_updates':cfg['source_sft_updates']-513,
                                  'reference':'B' if cfg['repair_arm']=='warm' else 'W'}}


def standard_counts(result):
    s=result['summary']
    return [s[m][k]['successes'] for m,k in (('sample','complete'),('sample','recovery'),('greedy','complete'),('greedy','recovery'))]


def target_counts(result):
    return [result['targeted_summary'][m]['successes'] for m in ('sample','greedy')]


def b_gate(w,b):
    a,c=standard_counts(w),standard_counts(b)
    first=b['targeted_summary']['sample']['first_action_certifications']
    return {'passed':all(y>=x for x,y in zip(a,c)) and b['evaluation_quality_passed'] and
                     first>=3 and first>w['targeted_summary']['sample']['first_action_certifications'],
            'first_action_certifications':first,'minimum':3}


def candidate_checks(w,c,b=None):
    a,d=standard_counts(w),standard_counts(c);ta,td=target_counts(w),target_counts(c)
    primary=1 if a[0]==16 else 0
    checks={'standard_non_regression':all(y>=x for x,y in zip(a,d)),
            'primary_gain':d[primary]>a[primary],'compound_sample_gain':td[0]>=ta[0]+2,
            'compound_greedy_non_regression':td[1]>=ta[1],
            'quality':bool(c['evaluation_quality_passed'])}
    if b is not None:
        bs=standard_counts(b);bt=target_counts(b)
        checks.update(standard_non_regression_vs_b=all(y>=x for x,y in zip(bs,d)),
                      grpo_increment=any(y>x for x,y in zip(bs+bt,d+td)))
    return {'passed':all(checks.values()),'checks':checks,'primary_index':primary}
