"""Pinned inputs and native-record validation for the second rollout policy."""

from dataclasses import asdict

from collect_vllm_rollout import ROOT, read, inventory
from freeze_sft_baseline import verify as verify_baseline
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.rollouts.phase5_signal import select_cases
from agentic_post_training.rollouts.records import RolloutSample, pack_rollout_groups
from agentic_post_training.rewards.audit import audit_records
from agentic_post_training.trainers.policy_iteration import file_hash, verify_parent, seed_for

PLAN_PATH = ROOT / 'configs/grpo_policy_iteration.json'


def setup():
    from transformers import AutoTokenizer

    plan = read(PLAN_PATH)
    fixed = {'contract_version': 'grpo-policy-iteration-v1', 'iteration': 2,
             'optimizer_mode': 'carry_parent_adam', 'base_seed': 202610280,
             'seed_group_stride': 1000, 'group_size': 8, 'chunk_requests': 64,
             'learning_rate': 1e-5, 'beta': .02, 'clip_epsilon': .2, 'tis_cap': 2.,
             'max_grad_norm': 1., 'stop_sampled_k3': .05, 'stop_ratio_clip_fraction': .5,
             'max_logical_steps': 36, 'checkpoint_step': 1, 'resume_probe_step': 2,
             'old_new_logp_tolerance': 1e-5, 'reload_logp_tolerance': 1e-5,
             'max_wall_seconds': 1800, 'max_run_cost_yuan': 10.,
             'auto_promote_baseline': False, 'seed': 20261028}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('fixed next-iteration contract differs')
    parent, _, parent_manifest = verify_parent(ROOT, plan)
    if plan['parent_run'] != 'artifacts/checkpoints/grpo-first-20261027-01':
        raise ValueError('first GRPO parent required')
    baseline = ROOT / 'artifacts/baselines/sft-v2-phase3'
    if file_hash(baseline / 'manifest.json') != plan['baseline_manifest_sha256']:
        raise ValueError('reference baseline identity differs')
    verify_baseline(baseline)
    manifest = read(baseline / 'manifest.json')
    if (manifest['resolved_revision'] != '12fd25f77366fa6b3b4b768ec3050bf629380bac'
            or manifest['model'] != 'HuggingFaceTB/SmolLM2-135M-Instruct'):
        raise ValueError('pinned base revision differs')
    base = (ROOT / manifest['base_cache_files'][0]['source']).parent
    tok = AutoTokenizer.from_pretrained(str(baseline / 'checkpoint/tokenizer'),
                                       local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    original_signal = read(ROOT / 'configs/phase5_rollout_signal.json')
    original_update = read(ROOT / 'configs/grpo_first_update.json')
    for key, old in [('case_ids', original_signal),
                     ('training_train_diagnostic_case_ids', original_update),
                     ('evaluation_development_case_ids', original_update)]:
        if plan[key] != old[key]:
            raise ValueError('frozen training/evaluation input IDs differ')
    folder = ROOT / 'data/preferences/dpo_v2'
    data_manifest = read(folder / 'manifest.json')
    if data_manifest['baseline_manifest_sha256'] != plan['baseline_manifest_sha256']:
        raise ValueError('data baseline differs')
    splits = load_preference_splits(folder, data_manifest)
    train = {r.id: r.target() for r in splits['train']}
    dev = {r.id: r.target() for r in splits['development']}
    cases = select_cases(splits['train'], plan['case_ids'])
    if (plan['evaluation_development_case_ids'] != list(dev)
            or {(c.system, c.prompt) for c in train.values()} & {(c.system, c.prompt) for c in dev.values()}):
        raise ValueError('development identity or train/dev isolation differs')
    suites = {'train_diagnostic': [train[k] for k in plan['training_train_diagnostic_case_ids']],
              'development': [dev[k] for k in plan['evaluation_development_case_ids']]}
    encode = lambda c: tok.apply_chat_template(
        [{'role': 'system', 'content': c.system}, {'role': 'user', 'content': c.prompt}],
        tokenize=True, add_generation_prompt=True)
    prompts = [encode(c) for c in cases]
    if any(len(encode(c)) + 64 > 512 for c in cases + sum(suites.values(), [])):
        raise ValueError('context budget exceeded')
    old_seeds = {20261021 + g * 1000 + i for g in range(144) for i in range(8)}
    seeds = {seed_for(g, i) for g in range(144) for i in range(8)}
    if len(seeds) != 1152 or seeds & old_seeds:
        raise ValueError('new sample seed collision')
    config = read(ROOT / 'configs/rollout_vllm_smoke.json')
    if parent_manifest['device'] != 'NVIDIA L20':
        raise ValueError('this iteration is pinned to the verified L20 parent')
    return config, plan, baseline, base, tok, cases, prompts


def evaluation_suites(plan):
    folder = ROOT / 'data/preferences/dpo_v2'
    splits = load_preference_splits(folder, read(folder / 'manifest.json'))
    train = {r.id: r.target() for r in splits['train']}
    dev = {r.id: r.target() for r in splits['development']}
    return {'train_diagnostic': [train[k] for k in plan['training_train_diagnostic_case_ids']],
            'development': [dev[k] for k in plan['evaluation_development_case_ids']]}


def check_device(config):
    from collect_vllm_rollout import check_gpu
    device = check_gpu(config)
    if device != 'NVIDIA L20':
        raise ValueError('same NVIDIA L20 GPU required for this pinned iteration')
    return device


def identity(config, plan, baseline):
    inv = inventory(config, baseline)
    inv['source_sha256'] = {k.replace('\\', '/'): v for k, v in inv['source_sha256'].items()}
    files = [*sorted((ROOT / 'src/agentic_post_training').rglob('*.py')),
             *sorted((ROOT / 'configs').glob('*.json')),
             *sorted((ROOT / 'data/preferences/dpo_v2').glob('*')),
             *(ROOT / 'scripts' / name for name in (
                 'iteration_common.py', 'collect_grpo_iteration.py', 'train_grpo_iteration.py', 'run_grpo_iteration.py',
                 'train_grpo_first.py', 'collect_phase5_signal.py', 'collect_vllm_rollout.py',
                 'replay_cloud_smoke.py', 'verify_cloud_bundle.py', 'freeze_sft_baseline.py'))]
    inv['source_sha256'].update({p.relative_to(ROOT).as_posix(): file_hash(p) for p in files if p.is_file()})
    inv.update(iteration_plan=plan, policy_version={
        'sampler_and_old': plan['parent_parameter_sha256'],
        'parent_summary_sha256': plan['parent_summary_sha256'],
        'parent_adapter_sha256': plan['parent_adapter_sha256'],
        'reference': read(ROOT / 'configs/trainer_probability_contract.json')['frozen_fp32_parameter_sha256'],
        'parent_optimizer_sha256': plan['parent_optimizer_sha256'],
        'optimizer_mode': plan['optimizer_mode']},
        parent_manifest_sha256=file_hash(ROOT / plan['parent_run'] / 'run_manifest.json'))
    return inv


def validate_records(rows, cases, prompts, tokenizer):
    if len(rows) != len(cases) * 8 or len(prompts) != len(cases):
        raise ValueError('all eight new samples per input required')
    groups = []
    for g, case in enumerate(cases):
        group = []
        for i, row in enumerate(rows[g * 8:(g + 1) * 8]):
            s = RolloutSample(**row['sample'])
            if (row['case'] != asdict(case) or s.group_id != case.id or s.case_id != case.id
                    or s.sample_index != i or s.seed != seed_for(g, i) or s.prompt_token_ids != prompts[g]
                    or s.policy_token_log_probs != s.behavior_token_log_probs
                    or s.eos_token_id != tokenizer.eos_token_id or len(s.response_token_ids) > 64
                    or (s.finish_reason == 'length' and len(s.response_token_ids) != 64)
                    or s.text != tokenizer.decode(s.response_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    or s.raw_text != tokenizer.decode(s.response_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)):
                raise ValueError('new policy native sample identity or decode differs')
            group.append(s)
        groups.append(group)
    return groups, pack_rollout_groups(groups, tokenizer.pad_token_id), audit_records(rows)
