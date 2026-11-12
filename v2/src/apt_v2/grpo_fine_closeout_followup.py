"""Frozen three-policy follow-up for fine GRPO closeout credit."""
from .grpo_credit import CLOSEOUT, LOCAL
from .grpo_fine_closeout import W_SHA
from .grpo_repair_study import RepairTask

SCHEMA = 'fine_closeout_followup_v1'
RUN_ID = 'grpo-fine-closeout-followup-v1-20261112'
CAP = 21600
TASK_COUNT = 12
GAIN = 6
NAMES = ('w', 'local', 'closeout')
CONFIG = 'configs/grpo-fine-closeout-followup-v1.json'
TEMPLATE = 'configs/grpo-fine-closeout-followup-branch-v1.json'
INPUT = 'artifacts/inputs/grpo-fine-closeout-followup-20261112'
MANIFEST = 'grpo_fine_closeout_followup_release_manifest.json'
CONTRACT = 'reports/grpo-fine-closeout-followup-contract-v1-20261112.md'
VALIDATION = 'reports/grpo-fine-closeout-followup-validation-v1-20261112.json'
SOURCE_REVIEW_SHA = 'c8c770396a726a0404dffea26c2173b33e31eeceeeb703a3ef906436ca11ff26'

POLICIES = {
    'w': {'checkpoint': 'initial', 'version': 0, 'updates': 0, 'adapter_sha256': W_SHA},
    'local': {'checkpoint': 'checkpoints/group-0046', 'version': 43, 'updates': 43,
              'adapter_sha256': 'eb78febd0fd2cce0cdac6a930894d59eb07cada9b320ecaf1249f1932f9f708e'},
    'closeout': {'checkpoint': 'checkpoints/group-0047', 'version': 47, 'updates': 47,
                 'adapter_sha256': '1c209b708c7cdf334f6ec103e0f5974193078e9d6f3acc8773654b37faec2b5a'},
}


def master_config():
    return {'schema': SCHEMA, 'run_id': RUN_ID, 'max_wall_seconds': CAP, 'task_count': TASK_COUNT,
            'minimum_credit_recovery_gain': GAIN, 'branch_template': TEMPLATE, 'input_dir': INPUT,
            'contract_report': CONTRACT, 'validation_report': VALIDATION}


def validate_master(cfg):
    if cfg != master_config():
        raise ValueError('frozen three-policy follow-up contract differs')


def validate_branch(cfg):
    role = cfg['run_id'].removeprefix(RUN_ID+'-')
    method = CLOSEOUT if role == 'closeout' else LOCAL
    if (role not in ('template', *NAMES) or cfg['run_id'] != RUN_ID+'-'+role
            or cfg.get('repair_profile') != SCHEMA or cfg['repair_arm'] != 'plain'
            or cfg['credit_method'] != method or cfg.get('verifier_version') != 'sql_semantics_v2'
            or cfg['max_wall_seconds'] != CAP or cfg['formal_gpu_budget_seconds'] != CAP
            or cfg['source_sft_updates'] != 513 or cfg['source_dpo_updates'] != 11
            or cfg['baseline_W_sha256'] != W_SHA or cfg['adapter_sha256'] != W_SHA
            or cfg['eval']['splits'] != {'development': TASK_COUNT}
            or cfg['sample_eval']['task_count'] != TASK_COUNT
            or cfg['targeted_evaluation']['samples_per_context'] != 3
            or cfg['closing_evaluation'] != {'enabled':False}
            or cfg['max_groups'] != 48 or cfg['max_updates'] != 48
            or cfg['group_evaluation_cursors'] != [48] or cfg['minimum_actual_updates'] != 0):
        raise ValueError('frozen follow-up branch differs')


def panel_tasks():
    from .grpo_curriculum import month_pair
    result=[]
    for i,month in enumerate((3,6,10,12)*3):
        current,prior=month_pair(2048+i//4,month)
        seed=86000+i
        result.append(RepairTask(f'fine-closeout-followup-{i:03}','development',f'fine-closeout-family-{seed}',
                                 seed,current,prior,('Central','East','West','South')[i%4],
                                 f'FineCloseoutSegment-{seed}','transient_query' if i%2 else 'none'))
    return result


def validate_evidence(evidence):
    receipt=evidence['receipt']
    if (receipt['sha256']!=SOURCE_REVIEW_SHA or not receipt['budget_passed']
            or evidence['scope']!='local_returned_files' or not evidence['passed']
            or evidence['source_commit']!='4eeaf8d'
            or evidence['source_run_id']!='grpo-fine-closeout-credit-v1-20261112'
            or evidence['final_policies']!={n:POLICIES[n] for n in ('local','closeout')}
            or evidence['actual_updates']!={n:POLICIES[n]['updates'] for n in ('local','closeout')}):
        raise ValueError('reviewed checkpoint lineage differs')
