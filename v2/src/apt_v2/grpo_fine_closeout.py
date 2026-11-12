"""Matched fine GRPO comparison of action credit on verified SQL closeout."""
from .dpo import action_from_text
from .environment import Environment
from .grpo_closeout_study import _examples
from .grpo_credit import CLOSEOUT, LOCAL
from .grpo_repair_study import RepairTask, STANDARD, W_SHA, standard_counts, training_tasks

SCHEMA = 'fine_closeout_credit_v1'
PROFILE = SCHEMA
RUN_ID = 'grpo-fine-closeout-credit-v1-20261112'
MANIFEST = 'grpo_fine_closeout_credit_release_manifest.json'
CONFIG = 'configs/grpo-fine-closeout-credit-v1.json'
TEMPLATE = 'configs/grpo-fine-closeout-credit-branch-v1.json'
CONTRACT = 'reports/grpo-fine-closeout-credit-contract-v1-20261112.md'
VALIDATION = 'reports/grpo-fine-closeout-credit-validation-v1-20261112.json'
BRANCHES = ('local', 'closeout')
METHODS = {'local': LOCAL, 'closeout': CLOSEOUT}
STATES = ('credit_fresh_check', 'credit_calculate', 'credit_checked_submit')
CAP = 28800
GROUPS = 48
BLOCK = 4
PREFLIGHT_CAP = 1500
FINAL_RESERVE = 8100


def master_config():
    return {'schema': SCHEMA, 'run_id': RUN_ID, 'max_wall_seconds': CAP, 'max_groups': GROUPS,
            'preflight_seconds': PREFLIGHT_CAP, 'preflight_seed': 2026111257,
            'final_reserve_seconds': FINAL_RESERVE, 'branch_template': TEMPLATE,
            'contract_report': CONTRACT, 'validation_report': VALIDATION}


def validate_master(cfg):
    if cfg != master_config():
        raise ValueError('frozen fine closeout master differs')


def validate_branch(cfg):
    role = cfg['run_id'].removeprefix(RUN_ID+'-')
    roles = ('template', 'w', *BRANCHES, 'confirm-w', 'confirm-local', 'confirm-closeout')
    expected_method = CLOSEOUT if role in ('closeout', 'confirm-closeout') else LOCAL
    retained = cfg.get('closeout_states', [])
    if (cfg['run_id'] != RUN_ID+'-'+role or role not in roles
            or cfg.get('repair_profile') != PROFILE
            or cfg.get('repair_arm') != 'plain' or cfg['credit_method'] != expected_method
            or cfg['max_groups'] != GROUPS or cfg['max_updates'] != GROUPS
            or cfg['group_evaluation_cursors'] != [GROUPS]
            or cfg['max_wall_seconds'] != CAP or cfg['formal_gpu_budget_seconds'] != CAP
            or cfg['source_sft_updates'] != 513 or cfg['source_dpo_updates'] != 11
            or cfg['adapter_sha256'] != W_SHA or cfg['baseline_W_sha256'] != W_SHA
            or cfg.get('verifier_version') != 'sql_semantics_v2'
            or cfg['learning_rate'] != 1e-6 or cfg['k'] != 4
            or cfg['closing_evaluation'] != {'enabled': False}
            or cfg['minimum_actual_updates'] != 0
            or retained != [s for s in STATES if s in retained]
            or cfg.get('evaluation_panel') != ('confirmation' if role.startswith('confirm-') else 'development')):
        raise ValueError('frozen fine closeout branch differs')


def panel_tasks(panel):
    from .grpo_curriculum import month_pair
    first = 84000 if panel == 'development' else 85000
    tasks = []
    for i, month in enumerate((2, 5, 9, 11)):
        current, prior = month_pair((2040 if panel == 'development' else 2044)+i%2, month)
        n = first+i
        tasks.append(RepairTask(f'fine-closeout-{panel}-{i:03}', 'development',
                                f'fine-closeout-family-{n}', n, current, prior,
                                ('Central', 'East', 'West', 'South')[i],
                                f'FineCloseoutSegment-{n}', 'transient_query' if i%2 else 'none'))
    return tasks


def context(task, scenario, allowed_split='train'):
    if scenario not in STATES or allowed_split != 'train' or task.split != 'train':
        raise ValueError('closeout curriculum prefixes are train-only')
    if scenario == 'credit_calculate':
        from .dpo_recovery import build_context
        return build_context(task, 'stale_evidence_repair', allowed_split)
    index = next(i for i, t in enumerate(training_tasks()) if t.id == task.id)
    checked, submitted = _examples(task, index)
    example = checked if scenario == 'credit_fresh_check' else submitted
    env = Environment(task)
    records = []
    try:
        for message in example['messages'][2::2]:
            text = message['content']
            records.append({'text': text, 'row': env.step(action_from_text(text))})
        if env.done:
            raise ValueError('closeout train prefix exhausted')
        row = env.step(action_from_text(example['response']))
        if row['observation'] != example['observation']:
            raise ValueError('closeout expert replay differs')
        return {'context_id': f'{task.id}:{scenario}', 'task_id': task.id, 'scenario': scenario,
                'records': records, 'messages': example['messages'], 'chosen': example['response'],
                'chosen_observation': row['observation']}
    finally:
        env.close()


def start_spec(cfg, index):
    if type(index) is not int or not 0 <= index <= GROUPS:
        raise ValueError('fine closeout group cursor out of range')
    tasks = training_tasks()
    block, slot = divmod(index, BLOCK)
    scenario = None
    if slot == 1:
        scenario = STANDARD[block % len(STANDARD)]
    elif slot == 3:
        state = STATES[block % len(STATES)]
        scenario = state if state in cfg['closeout_states'] else STANDARD[(block+3) % len(STANDARD)]
    task_index = (block+6) % len(tasks) if slot == 2 else block % len(tasks)
    return {'index': index, 'task_id': tasks[task_index].id, 'scenario': scenario,
            'kind': 'recovery' if scenario else 'complete', 'seed': cfg['seed']+index*100000}


def diagnostic_plan(cfg):
    validate_master(cfg)
    tasks = training_tasks()
    return [{'index': i, 'task_id': tasks[i].id, 'scenario': STATES[i%3],
             'seed': cfg['preflight_seed']+i*100000} for i in range(12)]


def progressed(episode, scenario):
    if scenario == 'credit_calculate':
        return any('change_cents' in row['new_verified_metrics'] for row in episode['turns'])
    if scenario == 'credit_checked_submit':
        return episode['success']
    return episode['success'] or any(row['action'].get('tool') == 'check_result'
                                     and row['observation'].get('passed') is True and not row['done']
                                     for row in episode['turns'])


def preflight_result(groups):
    summaries = {}
    for state in STATES:
        rows = [g for g in groups if g['spec']['scenario'] == state]
        advance = [[progressed(e, state) for e in g['episodes']] for g in rows]
        summaries[state] = {'committed_groups': len(rows), 'mixed_groups': sum(0 < sum(a) < 4 for a in advance),
                            'progressing_episodes': sum(sum(a) for a in advance)}
    retained = [s for s in STATES if summaries[s]['committed_groups'] == 4 and summaries[s]['mixed_groups'] >= 2]
    return {'states': summaries, 'retained': retained, 'completed_groups': len(groups),
            'status': 'complete' if len(groups) == 12 else 'partial_timeout'}


def selection(w, results, minimum_credit_gain=2):
    baseline = standard_counts(w)
    full_count = w['summary']['sample']['complete'].get('episodes',16)
    primary = 1 if baseline[0] == full_count else 0
    checks = {}
    for name, candidate in results.items():
        count = standard_counts(candidate)
        detail = {'non_regression': all(y >= x for x, y in zip(baseline, count)),
                  'primary_gain': count[primary] > baseline[primary],
                  'quality': w['evaluation_quality_passed'] and candidate['evaluation_quality_passed'],
                  'trained_adapter': candidate['spec']['policy']['updates'] > 0
                                     and candidate['spec']['policy']['adapter_sha256'] != W_SHA,
                  'effective_bf16_change': bool(candidate.get('effective_bf16_change'))}
        checks[name] = {'counts': count, 'checks': detail, 'passed': all(detail.values())}
    qualified = [name for name in results if checks[name]['passed']]
    chosen = max(qualified, key=lambda name: (checks[name]['counts'][primary], checks[name]['counts'][1],
                                               name == 'local')) if qualified else None
    gain = None
    if all(name in results for name in BRANCHES):
        gain = checks['closeout']['counts'][1] - checks['local']['counts'][1]
    return {'selected_candidate': chosen, 'formal_baseline': 'W', 'w_counts': baseline, 'checks': checks,
            'primary_metric': 'sample_complete' if primary == 0 else 'sample_recovery',
            'new_credit_recovery_gain': gain, 'credit_threshold': minimum_credit_gain,
            'credit_gain_passed': gain is not None and gain >= minimum_credit_gain}
