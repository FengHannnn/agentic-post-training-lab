"""Validate the fixed-r16 inputs with the native tokenizer and SQL verifier."""
import argparse
from collections import Counter
import ctypes
from pathlib import Path
import sys
import time


def validate(output, config=None):
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from grpo_common import ROOT, atomic_json, file_hash, read, validate_config
    from grpo_repair_worker import sft_order, supervision_for, verify_for
    from run_v2_grpo_dose import branch_config
    from grpo_dose_worker import protocol_for
    from apt_v2.grpo import digest
    from apt_v2 import grpo_dose_study as dose, grpo_closeout_study as closeout, grpo_lr_midpoint as midpoint
    from apt_v2.grpo_warmup_probe import make_plan, execute, productive, variants
    from apt_v2.sft import encode_example

    started = time.monotonic()
    cfg = read(config or ROOT/'configs/grpo-sql-lr-study-v1.json')
    lr = protocol_for(cfg)
    if lr is dose:
        raise ValueError('learning-rate master config required')
    lr.validate_master(cfg)
    folder = ROOT/cfg['input_dir']
    lr.validate_prior(read(folder/'prior-evidence.json'), cfg)
    template = read(ROOT/cfg['branch_template'])
    validate_config(template)
    branch_config(cfg, ROOT/'artifacts'/cfg['run_id'], 'w')
    identity = read(ROOT/template['initial_dir']/'model_identity.json')
    for name, sha in identity['files_sha256'].items():
        if not name.endswith('.safetensors') and file_hash(folder/'tokenizer'/name) != sha:
            raise ValueError('published tokenizer differs from W')
    tokenizer = AutoTokenizer.from_pretrained(folder/'tokenizer', local_files_only=True, trust_remote_code=False)

    data = read(ROOT/cfg['sft_examples'])
    reference = read(ROOT/cfg['probe_examples']) if 'probe_examples' in cfg else data
    plan = read(folder/'probe-plan.json')
    if plan != make_plan(reference):
        raise ValueError('published probe plan differs')
    old_master = read(ROOT/'configs/grpo-sql-dose-study-v1.json')
    old_recipe = dose.recipe(16)
    expected_batches = dose.order(dose.warm_config(old_master, 16), reference)
    maintenance_changes = []
    if lr is closeout:
        if len(data) != len(reference) or digest(reference) != closeout.R16_DATASET_SHA:
            raise ValueError('closeout requires the original r16 reference')
        for old, new in zip(reference, data):
            if any(old[k] != new[k] for k in ('task_id', 'state', 'role', 'split')):
                raise ValueError('closeout must preserve supervision slots')
            if (old != new) != (old['state'] in closeout.REPLACED_STATES):
                raise ValueError('only the designated maintenance slots may change')
            if old != new:
                maintenance_changes.append(old['state'])
        if Counter(maintenance_changes) != {state: 12 for state in closeout.REPLACED_STATES}:
            raise ValueError('closeout must replace both states on all train tasks')
    arms = {}
    for name, rate in tqdm(lr.CANDIDATES.items(), desc='Validate learning-rate arms', unit='candidate',
                           file=sys.stdout, disable=False):
        warm = lr.warm_config(cfg, rate)
        if data != supervision_for(warm):
            raise ValueError('published supervision differs from the registered dataset')
        verify_for(warm, data)
        batches = sft_order(warm)
        counts = Counter(i for batch in batches for i in batch)
        if batches != expected_batches or any(counts[i] != (16 if e['role'] == 'target' else 1)
                                              for i, e in enumerate(data)):
            raise ValueError('learning-rate study requires the identical r16 exposures and order')
        if lr is closeout:
            if warm['sft'] != {**midpoint.recipe(rate), 'supervision_sha256': digest(data)}:
                raise ValueError('closeout must preserve the optimizer recipe and bind the new data')
        elif {k for k in old_recipe if warm['sft'][k] != old_recipe[k]} != {'learning_rate'}:
            raise ValueError('learning rate must be the sole recipe change')
        arms[name] = {'learning_rate': rate, 'steps': len(batches), 'target_exposures': 192,
                      'maintenance_exposures': 48, 'source_sft_updates': 573, 'order_sha256': digest(batches)}

    max_prompt = max_response = 0
    for example in tqdm(data, desc='Encode fixed supervision', unit='action', file=sys.stdout, disable=False):
        encoded = encode_example(tokenizer, example, 8192)
        max_prompt = max(max_prompt, encoded['prompt_length'])
        max_response = max(max_response, encoded['response_length'])
        if encoded['response_length'] > 256:
            raise ValueError('native response exceeds the generation cap')
    for item in tqdm(plan, desc='Replay native SQL probes', unit='context', file=sys.stdout, disable=False):
        if not productive(item, execute(item, item['response'])):
            raise ValueError('expert probe action failed')
        if item['kind'] == 'target' and productive(item, execute(item, variants(tokenizer, item)['omitted_text'])):
            raise ValueError('missing-alias counterfactual certified')

    result = {'passed': True, 'dataset_sha256': digest(data), 'candidates': arms,
              'native_supervision_actions': len(data), 'probe_contexts_per_model': len(plan),
              'maximum_prompt_tokens': max_prompt, 'maximum_response_tokens': max_response,
              'prior_elapsed_seconds': cfg['prior_elapsed_seconds'], 'max_wall_seconds': cfg['max_wall_seconds'],
              'total_authorized_seconds': cfg['total_authorized_seconds'], 'run_grpo': False,
              'elapsed_seconds': time.monotonic()-started}
    if lr is closeout:
        result['reference_dataset_sha256'] = digest(reference)
        result['maintenance_replacements'] = dict(Counter(maintenance_changes))
        result['probe_plan_unchanged'] = True
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--sqlite-dll', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.sqlite_dll:
        sqlite_runtime = ctypes.WinDLL(str(args.sqlite_dll.resolve()))
    import sqlite3
    if sqlite3.sqlite_version != '3.54.1':
        raise ValueError('SQLite 3.54.1 required')
    validate(args.output, args.config)
