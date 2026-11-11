"""Single-rate midpoint of the fixed-r16 warm-start comparison."""
from .grpo_dose_study import recipe as dose_recipe, select_warm_start
from .grpo_lr_study import (
    TOTAL_SECONDS, master_contract, validate_prior as validate_lr_prior, verify_r16_order,
)

SCHEMA = 'repair_lr_midpoint_v1'
WARM_SCHEMA = 'repair_lr_midpoint_warm_v1'
RUN_ID = 'grpo-sql-lr-midpoint-v1-20261111'
MANIFEST = 'grpo_sql_lr_midpoint_release_manifest.json'
CANDIDATES = {'lr4': 4e-6}
INPUT_DIR = 'artifacts/inputs/grpo-sql-lr-midpoint-20261111'
MAX_WALL_SECONDS = 13500
PRIOR_SECONDS = 17842.66665698681

METRICS = {'current_net_cents', 'prior_net_cents', 'change_cents'}


def recipe(learning_rate):
    if learning_rate != CANDIDATES['lr4']:
        raise ValueError('midpoint requires the frozen 4e-6 learning rate')
    return {**dose_recipe(16), 'learning_rate': learning_rate}


def validate_master(cfg):
    expected = {
        **master_contract(),
        'schema': SCHEMA,
        'run_id': RUN_ID,
        'max_wall_seconds': MAX_WALL_SECONDS,
        'prior_elapsed_seconds': PRIOR_SECONDS,
        'learning_rates': list(CANDIDATES.values()),
        'branch_template': 'configs/grpo-sql-lr-midpoint-branch-v1.json',
        'input_dir': INPUT_DIR,
        'sft_examples': INPUT_DIR + '/sft-examples.json',
        'contract_report': 'reports/grpo-sql-lr-midpoint-contract-v1-20261111.md',
        'preparation_report': 'reports/grpo-sql-lr-midpoint-preparation-v1-20261111.md',
        'validation_report': 'reports/grpo-sql-lr-midpoint-validation-v1-20261111.json',
    }
    if cfg != expected or cfg['run_grpo'] is not False or cfg['stop_after_first_full_gate_pass'] is not True:
        raise ValueError('frozen midpoint study contract differs')
    if PRIOR_SECONDS + MAX_WALL_SECONDS > TOTAL_SECONDS:
        raise ValueError('midpoint exceeds remaining authorization')


def warm_config(cfg, learning_rate):
    validate_master(cfg)
    return {**cfg, 'schema': WARM_SCHEMA, 'run_id': f'{RUN_ID}-lr4', 'sft': recipe(learning_rate)}


def validate_warm(cfg):
    master = {key: value for key, value in cfg.items() if key != 'sft'}
    master.update(schema=SCHEMA, run_id=RUN_ID)
    if cfg != warm_config(master, cfg['sft']['learning_rate']):
        raise ValueError('frozen midpoint warm recipe differs')


def order(cfg, data, batches):
    validate_warm(cfg)
    return verify_r16_order(data, batches)


def validate_prior(data, cfg):
    receipts, audits = data['receipts'], data['local_audits']
    if len(receipts) != 5 or len(audits) != 5:
        raise ValueError('five prior receipts and audits required')
    validate_lr_prior({'receipts': receipts[:4], 'local_audits': audits[:4]}, master_contract())
    receipt, audit = receipts[-1], audits[-1]
    if (receipt['sha256'] != '6f64d527d9b931ab7387aebf3d26ac0de036d6531cdbebb87938f818259c480a'
            or receipt['combined_budget_passed'] is not True
            or receipt['combined_elapsed_seconds'] != cfg['prior_elapsed_seconds']
            or receipt['total_authorized_seconds'] != cfg['total_authorized_seconds']
            or sum(r['elapsed_seconds_including_packaging'] for r in receipts) != cfg['prior_elapsed_seconds']
            or audit['passed'] is not True or audit['actual_sft_updates'] != {'lr3': 60, 'lr5': 60}
            or audit['actual_rl_updates'] != 0
            or audit['selection']['stop_reason'] != 'no_qualified_warm_start'
            or audit['selection']['selected_warm_start'] is not None):
        raise ValueError('prior learning-rate result or authorization ledger differs')


def candidate_fields(name):
    return {'name': name, 'dose': 16, 'learning_rate': CANDIDATES[name]}


def selection(w, records):
    return select_warm_start(w, records, CANDIDATES, candidate_fields)


def _completion_rows(result):
    rows = {}
    for episode in result['episodes']:
        key = tuple(episode[k] for k in ('task_id', 'scenario', 'evaluation_mode', 'rollout_index', 'seed'))
        if key in rows:
            raise ValueError('duplicate standard episode')
        prefix = [item['row'] for item in episode['start']['prefix']]
        metrics = {metric for row in [*prefix, *episode['turns']] for metric in row['new_verified_metrics']}
        checks = [turn for turn in episode['turns'] if (turn['action'] or {}).get('tool') == 'check_result']
        submits = [turn for turn in episode['turns'] if (turn['action'] or {}).get('tool') == 'submit']
        checked = any(turn['observation'].get('passed') is True for turn in checks)
        submitted = any(turn['observation'].get('submitted') is True for turn in submits)
        if submitted != episode['success']:
            raise ValueError('standard episode submit differs from terminal success')
        rows[key] = {
            'start_id': episode['start_id'],
            'sampling': episode['sampling'],
            'all_metrics_certified': METRICS <= metrics,
            'check_passed': checked,
            'submit_attempted': bool(submits),
            'submitted_successfully': submitted,
            'budget_exhausted': episode['termination'] == 'budget',
        }
    return rows


def completion_diagnostics(w, candidate):
    """Track standard task stages, using the same W/candidate episode pairs."""
    models = {'w': _completion_rows(w)}
    if candidate is not None:
        models['lr4'] = _completion_rows(candidate)
        left, right = models['w'], models['lr4']
        if left.keys() != right.keys() or any(
            (row['start_id'], row['sampling']) != (right[key]['start_id'], right[key]['sampling'])
            for key, row in left.items()
        ):
            raise ValueError('standard evaluation pair differs')

    summary = {}
    paired = {} if candidate is not None else None
    for mode in ('sample', 'greedy'):
        for kind, complete in (('complete', True), ('recovery', False)):
            group = f'{mode}_{kind}'
            keys = [key for key in models['w'] if key[2] == mode and (key[1] is None) == complete]
            summary[group] = {}
            for name, rows in models.items():
                group_rows = [rows[key] for key in keys]
                summary[group][name] = {
                    'episodes': len(group_rows),
                    'all_metrics_certified': sum(row['all_metrics_certified'] for row in group_rows),
                    'check_passed': sum(row['check_passed'] for row in group_rows),
                    'submit_attempted': sum(row['submit_attempted'] for row in group_rows),
                    'submitted_successfully': sum(row['submitted_successfully'] for row in group_rows),
                    'all_metrics_without_submit': sum(row['all_metrics_certified'] and not row['submit_attempted']
                                                      for row in group_rows),
                    'passed_check_without_submit': sum(row['check_passed'] and not row['submit_attempted']
                                                       for row in group_rows),
                }
            if paired is not None:
                losses = [right[key] for key in keys
                          if left[key]['submitted_successfully'] and not right[key]['submitted_successfully']]
                paired[group] = {
                    'w_success_candidate_failure': len(losses),
                    'w_failure_candidate_success': sum(
                        not left[key]['submitted_successfully'] and right[key]['submitted_successfully'] for key in keys
                    ),
                    'candidate_losses_with_all_metrics': sum(row['all_metrics_certified'] for row in losses),
                    'candidate_losses_with_passed_check': sum(row['check_passed'] for row in losses),
                    'candidate_losses_without_submit': sum(not row['submit_attempted'] for row in losses),
                    'candidate_losses_exhausting_budget': sum(row['budget_exhausted'] for row in losses),
                }
    return {'standard': summary, 'paired': paired}
