"""Fixed-r16 warm-start search with learning rate as the sole treatment."""
from .grpo import digest
from .grpo_dose_study import recipe as dose_recipe, select_warm_start, validate_prior as validate_dose_prior

SCHEMA = 'repair_lr_v1'
WARM_SCHEMA = 'repair_lr_warm_v1'
RUN_ID = 'grpo-sql-lr-study-v1-20261111'
MANIFEST = 'grpo_sql_lr_release_manifest.json'
CANDIDATES = {'lr3': 3e-6, 'lr5': 5e-6}
INPUT_DIR = 'artifacts/inputs/grpo-sql-lr-20261111'
MAX_WALL_SECONDS = 18600
PRIOR_SECONDS = 12825.341599632986
TOTAL_SECONDS = 31527.179048960097
DATASET_SHA = '0e53fafba730aa7ad815457fc60d0a973e77a015774ea5c8e583abb38ba36dc2'
ORDER_SHA = 'fa141467d98e1814a6d88fa866eebc2adc434cab61430430a92e07b0d92d8e37'


def recipe(learning_rate):
    if learning_rate not in CANDIDATES.values():
        raise ValueError('frozen warm-up learning rate required')
    return {**dose_recipe(16), 'learning_rate': learning_rate}


def master_contract():
    return {
        'schema': SCHEMA,
        'run_id': RUN_ID,
        'max_wall_seconds': MAX_WALL_SECONDS,
        'new_authorized_seconds': 21600,
        'prior_elapsed_seconds': PRIOR_SECONDS,
        'total_authorized_seconds': TOTAL_SECONDS,
        'target_repeats': 16,
        'learning_rates': list(CANDIDATES.values()),
        'seed': 2026111061,
        'train_gate': {'minimum_target_per_repeat': 9, 'maintenance_non_regression': True,
                       'repeat_consistency_non_regression': True, 'effective_bf16_change': True},
        'stop_after_first_full_gate_pass': True,
        'run_grpo': False,
        'branch_template': 'configs/grpo-sql-lr-branch-v1.json',
        'input_dir': INPUT_DIR,
        'sft_examples': INPUT_DIR + '/sft-examples.json',
        'contract_report': 'reports/grpo-sql-lr-contract-v1-20261111.md',
        'preparation_report': 'reports/grpo-sql-lr-preparation-v1-20261111.md',
        'validation_report': 'reports/grpo-sql-lr-validation-v1-20261111.json',
        'sampling': {'temperature': 1, 'top_p': 1, 'top_k': -1, 'min_p': 0, 'repetition_penalty': 1,
                     'presence_penalty': 0, 'frequency_penalty': 0, 'min_tokens': 0, 'max_new_tokens': 256,
                     'max_model_len': 8192, 'logprobs_mode': 'processed_logprobs'},
    }


def validate_master(cfg):
    if cfg != master_contract() or cfg.get('run_grpo') is not False or cfg.get('stop_after_first_full_gate_pass') is not True:
        raise ValueError('frozen learning-rate study contract differs')
    if PRIOR_SECONDS + MAX_WALL_SECONDS > TOTAL_SECONDS:
        raise ValueError('learning-rate study exceeds remaining authorization')


def warm_config(cfg, learning_rate):
    validate_master(cfg)
    sft = recipe(learning_rate)
    name = next(name for name, rate in CANDIDATES.items() if rate == learning_rate)
    return {**cfg, 'schema': WARM_SCHEMA, 'run_id': f'{RUN_ID}-{name}', 'sft': sft}


def validate_warm(cfg):
    master = {key: value for key, value in cfg.items() if key != 'sft'}
    master.update(schema=SCHEMA, run_id=RUN_ID)
    if cfg != warm_config(master, cfg['sft']['learning_rate']):
        raise ValueError('frozen learning-rate warm recipe differs')


def order(cfg, data, batches):
    validate_warm(cfg)
    return verify_r16_order(data, batches)


def verify_r16_order(data, batches):
    if digest(data) != DATASET_SHA or digest(batches) != ORDER_SHA:
        raise ValueError('learning-rate arms require the exact published r16 data and order')
    return batches


def validate_prior(data, cfg):
    receipts, audits = data['receipts'], data['local_audits']
    if len(receipts) != 4 or len(audits) != 4:
        raise ValueError('four prior receipts and audits required')
    validate_dose_prior(
        {'receipts': receipts[:3], 'local_audits': audits[:3]},
        {'prior_elapsed_seconds': 9927.179048960097},
    )
    receipt, audit = receipts[-1], audits[-1]
    if (receipt['sha256'] != '88ed43afd7e9cdc88e63af1486339044f0fe037993b8582992c763155c44ab04'
            or not receipt['combined_budget_passed']
            or receipt['combined_elapsed_seconds'] != cfg['prior_elapsed_seconds']
            or receipt['total_authorized_seconds'] != cfg['total_authorized_seconds']
            or sum(r['elapsed_seconds_including_packaging'] for r in receipts) != cfg['prior_elapsed_seconds']
            or audit['passed'] is not True or audit['actual_sft_updates'] != {'r4': 24, 'r8': 36, 'r16': 60}
            or audit['actual_rl_updates'] != 0
            or audit['selection']['stop_reason'] != 'no_qualified_warm_start'
            or audit['selection']['selected_warm_start'] is not None):
        raise ValueError('dose study result or authorization ledger differs')


def candidate_fields(name):
    return {'name': name, 'dose': 16, 'learning_rate': CANDIDATES[name]}


def selection(w, records):
    return select_warm_start(w, records, CANDIDATES, candidate_fields)
