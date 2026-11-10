"""Pre-registered target-dose search with fixed maintenance, optimizer and ordered gates."""
import copy,random
from .grpo_repair_study import W_SHA,b_gate
from .grpo_retention_study import verify_dataset

SCHEMA='repair_dose_v1'
WARM_SCHEMA='repair_dose_warm_v1'
RUN_ID='grpo-sql-dose-study-v1-20261110'
DOSES=(4,8,16)


def recipe(repeats):
    if type(repeats) is not int or repeats not in DOSES:raise ValueError('frozen target dose required')
    return {'steps':12+3*repeats,'batch_size':4,'epochs':1,'learning_rate':2e-6,'seed':2026111053,
            'target_actions':12,'maintenance_actions':48,'target_repeats':repeats}


def validate_master(cfg):
    if (cfg.get('schema')!=SCHEMA or cfg.get('run_id')!=RUN_ID or cfg.get('target_repeats')!=list(DOSES)
            or cfg.get('max_wall_seconds')!=21600 or cfg.get('new_authorized_seconds')!=21600
            or cfg.get('prior_elapsed_seconds')!=9927.179048960097
            or cfg.get('total_authorized_seconds')!=31527.179048960097 or cfg.get('seed')!=2026111061
            or cfg.get('train_gate')!={'minimum_target_per_repeat':9,'maintenance_non_regression':True,
                                      'repeat_consistency_non_regression':True,'effective_bf16_change':True}
            or cfg.get('branch_template')!='configs/grpo-sql-dose-branch-v1.json'
            or cfg.get('input_dir')!='artifacts/inputs/grpo-sql-dose-20261110'
            or cfg.get('sft_examples')!='artifacts/inputs/grpo-sql-dose-20261110/sft-examples.json'
            or cfg.get('sampling')!={'temperature':1,'top_p':1,'top_k':-1,'min_p':0,'repetition_penalty':1,
                 'presence_penalty':0,'frequency_penalty':0,'min_tokens':0,'max_new_tokens':256,'max_model_len':8192,
                 'logprobs_mode':'processed_logprobs'}
            or cfg.get('stop_after_first_full_gate_pass') is not True or cfg.get('run_grpo') is not False):
        raise ValueError('frozen target-dose master contract differs')
    if cfg['prior_elapsed_seconds']+cfg['max_wall_seconds']!=cfg['total_authorized_seconds']:
        raise ValueError('new authorization budget ledger differs')


def warm_config(cfg,repeats):
    validate_master(cfg);out=copy.deepcopy(cfg)
    out.update(schema=WARM_SCHEMA,run_id=RUN_ID+'-r'+str(repeats),sft=recipe(repeats),dose=repeats)
    return out


def validate_warm(cfg):
    repeats=cfg.get('dose');expected=recipe(repeats);master=copy.deepcopy(cfg)
    master.update(schema=SCHEMA,run_id=RUN_ID);master.pop('dose',None);master.pop('sft',None)
    validate_master(master)
    if cfg['run_id']!=RUN_ID+'-r'+str(repeats) or cfg['schema']!=WARM_SCHEMA or cfg['sft']!=expected:
        raise ValueError('frozen dose SFT recipe differs')


def order(cfg,data):
    validate_warm(cfg);verify_dataset(data);r=cfg['dose']
    indices=[i for i,e in enumerate(data) for _ in range(r if e['role']=='target' else 1)]
    rng=random.Random(cfg['sft']['seed']);rng.shuffle(indices)
    if len(indices)!=cfg['sft']['steps']*4:raise ValueError('dose exposure count differs')
    return [indices[i:i+4] for i in range(0,len(indices),4)]


def validate_prior(data,cfg):
    expected=('77d7e4cea2c9d0d481ee95a20583d00e8cc27a992c35ae46d983179bc00b9920',
              '0246e5f7b2fad61fecddc684d64e459fc71da43aa48dca1c9b1ea61d7eb98a62',
              '69d0017d727cbbd5f041c34e649c43bca2ef56e7b835bbaaab03bb4bdb270604')
    receipts=data['receipts']
    if (tuple(r['sha256'] for r in receipts)!=expected or receipts[-1]['combined_elapsed_seconds']!=cfg['prior_elapsed_seconds']
            or sum(r['elapsed_seconds_including_packaging'] for r in receipts)!=cfg['prior_elapsed_seconds']
            or not all(a['passed'] for a in data['local_audits'])):
        raise ValueError('prior results/budget evidence differs')
    probe=data['local_audits'][-1]
    if [probe['models'][n]['target_first_certifications'] for n in ('w','b24','b15')]!=[0,11,0]:
        raise ValueError('prior target acquisition evidence differs')
    return True


def train_gate(w,b,changed):
    checks={'both_target_repeats_at_least_nine':all(n>=9 for n in b['target_certifications_by_repeat']),
            'both_target_repeats_above_W':all(y>x for x,y in zip(w['target_certifications_by_repeat'],b['target_certifications_by_repeat'])),
            'maintenance_non_regression':b['maintenance_productive_actions']>=w['maintenance_productive_actions'],
            'maintenance_each_state_non_regression':all(b['maintenance_by_state'][s]['productive']>=v['productive'] for s,v in w['maintenance_by_state'].items()),
            'repeat_consistency_non_regression':b['target_repeat_identical']>=w['target_repeat_identical'],
            'effective_bf16_change':bool(changed)}
    return {'passed':all(checks.values()),'checks':checks,'target_certifications_by_repeat':b['target_certifications_by_repeat']}


def selection(w,records):
    selected=None
    if not w['evaluation_quality_passed']:
        if records:raise ValueError('baseline quality rejection must precede candidate work')
        return {'selected_warm_start':None,'selected_candidate':None,'formal_baseline':'W','stop_reason':'baseline_quality_rejected'}
    for i,r in enumerate(records):
        if r['dose']!=DOSES[i]:raise ValueError('ordered dose search differs')
        if not r['train_gate']['passed']:
            if r.get('full_gate') is not None:raise ValueError('rejected train gate must skip full evaluation')
        elif r.get('full_gate') is None:raise ValueError('qualified train probe requires full gate')
        elif r['full_gate']['passed']:
            selected='r'+str(r['dose'])
            if i!=len(records)-1:raise ValueError('candidate work continued after first full gate pass')
    if selected is None and len(records)!=3:raise ValueError('complete search requires all three doses or a passing B')
    return {'selected_warm_start':selected,'selected_candidate':None,'formal_baseline':'W',
            'stop_reason':'warm_start_ready' if selected else 'no_qualified_warm_start','run_grpo':False}
