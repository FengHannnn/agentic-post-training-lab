"""Fixed-r16 maintenance ablation for verified evidence and submission."""
import copy
import json

from .dpo import TEMPORARY_ERROR, corrective_action
from .environment import Environment
from .grpo import digest
from .grpo_dose_study import select_warm_start
from .grpo_lr_midpoint import (
    INPUT_DIR as MIDPOINT_INPUT, PRIOR_SECONDS as MIDPOINT_PRIOR, recipe as midpoint_recipe,
    validate_prior as validate_midpoint_prior,
)
from .grpo_lr_study import DATASET_SHA as R16_DATASET_SHA, ORDER_SHA, TOTAL_SECONDS, master_contract
from .grpo_repair_study import training_tasks
from .grpo_retention_study import supervision_dataset as retention_dataset
from .runtime import SYSTEM
from .tasks import metric_sql

SCHEMA = 'repair_closeout_v1'
WARM_SCHEMA = 'repair_closeout_warm_v1'
RUN_ID = 'grpo-sql-closeout-v1-20261112'
MANIFEST = 'grpo_sql_closeout_release_manifest.json'
CANDIDATES = {'closeout': 4e-6}
COMPLETION_CANDIDATE = 'closeout'
INPUT_DIR = 'artifacts/inputs/grpo-sql-closeout-20261112'
PROBE_EXAMPLES = MIDPOINT_INPUT + '/sft-examples.json'
MAX_WALL_SECONDS = 9000
PRIOR_SECONDS = 22377.370386653813
REPLACED_STATES = ('failed_check_repair', 'checked_submit')
DATASET_SHA = 'bcb9908b6006705e5a756cdce4ad09d35295eff9658c22537b366c0428727027'


def _examples(task, index):
    env = Environment(task)
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': task.request()}]

    def step(action):
        row = env.step(action)
        messages.extend([{'role': 'assistant', 'content': json.dumps(action)},
                         {'role': 'user', 'content': json.dumps({'observation': row['observation']})}])
        return row

    def query(sql):
        action = {'tool': 'query_sql', 'arguments': {'sql': sql}}
        row = step(action)
        if row['tool_error'] == TEMPORARY_ERROR:
            row = step(action)
        if row['tool_error']:
            raise ValueError('closeout query failed')
        return row

    def sample(state, action):
        prompt = copy.deepcopy(messages)
        row = step(action)
        if row['tool_error'] or row['observation'].get('passed') is not True:
            raise ValueError('closeout action failed')
        return {'task_id': task.id, 'split': 'train', 'role': 'maintenance', 'state': state,
                'messages': prompt, 'response': json.dumps(action), 'observation': row['observation']}

    try:
        step(corrective_action(env))
        step(corrective_action(env))
        metrics = ('current_net_cents', 'prior_net_cents')
        stale = metrics[index % 2]
        other = metrics[1 - index % 2]
        stale_sql = metric_sql(task, stale)
        if index % 4 >= 2:
            query(metric_sql(task, other))
        old = query(stale_sql.rsplit(' AS ', 1)[0])
        if old['observation']['verified_metrics']:
            raise ValueError('old evidence must be unverified')
        if index % 4 < 2 or index % 3 == 0:
            query(metric_sql(task, other))
        fresh = query(stale_sql)
        if fresh['new_verified_metrics'] != [stale]:
            raise ValueError('fresh evidence must certify the missing metric')
        step(corrective_action(env))
        correct_check = corrective_action(env)
        fresh_id = fresh['observation']['evidence_id']
        old_id = old['observation']['evidence_id']
        if correct_check['tool'] != 'check_result' or correct_check['arguments']['evidence'][stale] != fresh_id:
            raise ValueError('expert check must reference the fresh evidence')
        wrong_check = copy.deepcopy(correct_check)
        wrong_check['arguments']['evidence'][stale] = old_id
        failed = step(wrong_check)['observation']
        if failed['passed'] or [k for k, ok in failed['checks'].items() if not ok] != [stale]:
            raise ValueError('stale reference must be the sole check failure')
        checked = sample('failed_check_repair', correct_check)
        submit = corrective_action(env)
        if submit != {'tool': 'submit', 'arguments': correct_check['arguments']}:
            raise ValueError('submit must preserve the passed check arguments')
        submitted = sample('checked_submit', submit)
        if not submitted['observation'].get('submitted') or not env.success:
            raise ValueError('closeout sequence must complete within the action budget')
        return checked, submitted
    finally:
        env.close()


def supervision_dataset():
    base = retention_dataset()
    if digest(base) != R16_DATASET_SHA:
        raise ValueError('original r16 supervision differs')
    replacements = {(e['task_id'], e['state']): e
                    for i, task in enumerate(training_tasks()) for e in _examples(task, i)}
    return [replacements.get((example['task_id'], example['state']), example) for example in base]


def recipe(learning_rate):
    return {**midpoint_recipe(learning_rate), 'supervision_sha256': DATASET_SHA}


def validate_master(cfg):
    expected = {
        **master_contract(),
        'schema': SCHEMA,
        'run_id': RUN_ID,
        'max_wall_seconds': MAX_WALL_SECONDS,
        'prior_elapsed_seconds': PRIOR_SECONDS,
        'learning_rates': list(CANDIDATES.values()),
        'branch_template': 'configs/grpo-sql-closeout-branch-v1.json',
        'input_dir': INPUT_DIR,
        'sft_examples': INPUT_DIR + '/sft-examples.json',
        'probe_examples': PROBE_EXAMPLES,
        'contract_report': 'reports/grpo-sql-closeout-contract-v1-20261112.md',
        'preparation_report': 'reports/grpo-sql-closeout-preparation-v1-20261112.md',
        'validation_report': 'reports/grpo-sql-closeout-validation-v1-20261112.json',
    }
    if cfg != expected or cfg['run_grpo'] is not False or cfg['stop_after_first_full_gate_pass'] is not True:
        raise ValueError('frozen closeout study contract differs')
    if PRIOR_SECONDS + MAX_WALL_SECONDS > TOTAL_SECONDS:
        raise ValueError('closeout exceeds remaining authorization')


def warm_config(cfg, learning_rate):
    validate_master(cfg)
    return {**cfg, 'schema': WARM_SCHEMA, 'run_id': f'{RUN_ID}-closeout', 'sft': recipe(learning_rate)}


def validate_warm(cfg):
    master = {key: value for key, value in cfg.items() if key != 'sft'}
    master.update(schema=SCHEMA, run_id=RUN_ID)
    if cfg != warm_config(master, cfg['sft']['learning_rate']):
        raise ValueError('frozen closeout warm recipe differs')


def order(cfg, data, batches):
    validate_warm(cfg)
    if digest(data) != DATASET_SHA or digest(batches) != ORDER_SHA:
        raise ValueError('closeout supervision or fixed r16 order differs')
    return batches


def validate_prior(data, cfg):
    receipts, audits = data['receipts'], data['local_audits']
    if len(receipts) != 6 or len(audits) != 6:
        raise ValueError('six prior receipts and audits required')
    validate_midpoint_prior(
        {'receipts': receipts[:5], 'local_audits': audits[:5]},
        {'prior_elapsed_seconds': MIDPOINT_PRIOR, 'total_authorized_seconds': TOTAL_SECONDS},
    )
    receipt, audit = receipts[-1], audits[-1]
    if (receipt['sha256'] != '76e1d9124a4ef7ec2ac3a682074e209299ab155d8dfb2bd10e803fd09c0bf453'
            or receipt['combined_budget_passed'] is not True
            or receipt['combined_elapsed_seconds'] != cfg['prior_elapsed_seconds']
            or receipt['total_authorized_seconds'] != cfg['total_authorized_seconds']
            or sum(r['elapsed_seconds_including_packaging'] for r in receipts) != cfg['prior_elapsed_seconds']
            or audit['passed'] is not True or audit['actual_sft_updates'] != {'lr4': 60}
            or audit['actual_rl_updates'] != 0
            or audit['selection']['stop_reason'] != 'no_qualified_warm_start'
            or audit['selection']['selected_warm_start'] is not None):
        raise ValueError('midpoint result or authorization ledger differs')


def candidate_fields(name):
    return {'name': name, 'dose': 16, 'learning_rate': CANDIDATES[name], 'maintenance_replacements': 24}


def selection(w, records):
    return select_warm_start(w, records, CANDIDATES, candidate_fields)
