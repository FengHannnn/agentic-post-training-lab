"""Recompute preflight groups, transitions, weight deltas and post-update refresh."""
import argparse
import math
import sys
from pathlib import Path
from grpo_common import (read, atomic_json, file_hash, verify_files, verify_inputs,
                         initial_policy, load_transition, under)
from grpo_worker import verify_group
from apt_v2.grpo import digest


def verified_zero_signal(cfg,state,groups,transitions,runner,changed_tensors):
    """An all-skipped alias run is evidence, with an explicit unchanged W policy."""
    return (cfg.get('curriculum_schema')=='alias_curriculum_v1' and state==initial_policy(cfg)
            and changed_tensors==0 and bool(transitions) and len(groups)==len(transitions)+1
            and runner['refresh_only_last_group'] and groups[-1]['policy']==state
            and all(not g['credit']['eligible'] for g in groups[:len(transitions)])
            and all(not r['updated'] and r['before']==state and r['after']==state
                    and r['metrics']['skip_reason']=='zero_variance' for r in transitions))


def audit(output, local=False):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm
    cfg = verify_inputs(output, runtime=not local, local=local)
    tok = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True, trust_remote_code=False)
    state = initial_policy(cfg)
    groups, transitions = [], []
    paths = sorted((output/'groups').glob('group-*.json'))
    expected_indices = list(range(len(paths)))
    if [int(p.stem[-4:]) for p in paths] != expected_indices:
        raise ValueError('contiguous sampled group schedule required')
    changed_tensors = 0
    for index in tqdm(expected_indices, desc='Strict GRPO replay', unit='group', file=sys.stdout, disable=False):
        g = verify_group(output, index, state, tok, local=local)
        if cfg.get('curriculum_schema')=='alias_curriculum_v1':
            from grpo_sampler_probe import METHOD
            if g['probability_probe'].get('verification_method')!=METHOD:
                raise ValueError('alias curriculum requires same-forward probability evidence')
        from grpo_vllm import projected_checkpoint_tensors
        projected = projected_checkpoint_tensors(under(output, state['checkpoint'])/'adapter')
        if any(p['tensors'] != projected or p['loaded_tensor_sha256'] != digest(projected) for p in g['worker_proof']):
            raise ValueError('independent training worker tensor projection differs')
        groups.append(g)
        path = output/'transitions'/f'group-{index:04}.json'
        if not path.exists():
            if index != len(paths)-1 or not read(output/'runner_state.json')['refresh_only_last_group']:
                raise ValueError('every training group needs an update/skip transaction')
            continue
        result = load_transition(output, index, state)
        if result['metrics']['credit'] != g['credit']:
            raise ValueError('update and sampling credit differ')
        if not result['updated']:
            if result['metrics']['skip_reason'] == 'zero_variance' and g['credit']['eligible']:
                raise ValueError('eligible group marked zero variance')
        else:
            before = under(output, state['checkpoint'])/'adapter/adapter_model.safetensors'
            after_dir = under(output, result['after']['checkpoint'])
            after = after_dir/'adapter/adapter_model.safetensors'
            a, b = load_file(before), load_file(after)
            if a.keys() != b.keys():
                raise ValueError('LoRA tensor keys changed')
            count = sum(not torch.equal(a[n], b[n]) for n in a)
            delta = sum((a[n].double()-b[n].double()).square().sum().item() for n in a)**.5
            m = result['metrics']
            if (count != m['changed_tensors'] or not math.isclose(delta, m['adapter_delta_l2'], abs_tol=1e-10, rel_tol=1e-7)
                    or delta <= 0 or m['frozen_parameter_gradients'] != 0 or not m['reference_unchanged']
                    or not m['optimizer_roundtrip_verified'] or not m['rng_roundtrip_verified']
                    or m['optimizer_steps'] != [result['after']['updates']]):
                raise ValueError('actual update, frozen parameters or saved state evidence differs')
            complete = read(after_dir/'complete.json')
            cache = read(after_dir/'scores.json')
            if (cache['group_sha256'] != digest(g['episodes']) or cache['policy_sha256'] != state['adapter_sha256']
                    or cache['reference_sha256'] != cfg['adapter_sha256'] or cache['same_path_max_error'] > 1e-5):
                raise ValueError('old/reference token cache binding differs')
            expected_entries = [(i, row) for i, e in enumerate(g['episodes']) for row in e['turns']]
            if len(cache['entries']) != len(expected_entries):
                raise ValueError('score cache action coverage differs')
            for entry, (i, row) in zip(cache['entries'], expected_entries):
                gen = row['generation']
                if (entry['episode'] != i or entry['turn'] != row['turn']
                        or entry['prompt_token_ids'] != gen['prompt_token_ids']
                        or entry['response_token_ids'] != gen['response_token_ids']
                        or entry['q'] != gen['behavior_token_log_probs']
                        or any(len(entry[k]) != len(entry['response_token_ids']) for k in ('old', 'reference', 'q'))
                        or any(not math.isfinite(x) for k in ('old', 'reference', 'q') for x in entry[k])):
                    raise ValueError('action score cache differs')
            if not local:
                if file_hash(after_dir/'state.pt') != complete['state_sha256']:
                    raise ValueError('saved Adam/RNG hash differs')
                saved = torch.load(after_dir/'state.pt', map_location='cpu', weights_only=False)
                if (saved['actual_updates'] != result['after']['updates'] or saved['group_cursor'] != index+1
                        or saved['policy_version'] != result['after']['version']
                        or any(int(v['step']) != saved['actual_updates'] for v in saved['optimizer']['state'].values())):
                    raise ValueError('cloud Adam/version/cursor differs')
            changed_tensors += count
        transitions.append(result)
        state = result['after']
    runner = read(output/'runner_state.json')
    if runner['policy'] != state or runner['next_group'] != len(transitions):
        raise ValueError('runner cursor or latest policy differs')
    refreshed = [g for g in groups if g['policy']['updates'] > 0]
    refresh_passed = any(g['policy'] == state for g in refreshed)
    zero_signal=verified_zero_signal(cfg,state,groups,transitions,runner,changed_tensors)
    if zero_signal:refresh_passed=True
    checks = {'minimum_actual_updates': state['updates'] >= cfg.get('minimum_actual_updates', cfg.get('minimum_preflight_updates', 1)),
              'post_update_sample_at_latest_version': refresh_passed,
              'all_worker_tensor_checks': all(p['passed'] for g in groups for p in g['worker_proof']),
              'all_probability_probes': all(g['probability_probe']['passed'] for g in groups),
              'actual_updated_weights': changed_tensors > 0,
              'group_cap': len(transitions) <= cfg['max_groups'], 'update_cap': state['updates'] <= cfg['max_updates']}
    if cfg.get('curriculum_schema')=='alias_curriculum_v1':
        checks['updated_or_verified_zero_signal']=checks.pop('actual_updated_weights') or zero_signal
    result = {'passed': all(checks.values()), 'checks': checks, 'scope': 'local_returned_files' if local else 'cloud_files_and_state',
              'actual_updates': state['updates'], 'verified_zero_signal': zero_signal, 'attempted_groups': len(transitions),
              'sampled_groups_including_refresh': len(groups),
              'eligible_training_groups': sum(g['credit']['eligible'] for g in groups[:len(transitions)]),
              'zero_variance_training_groups': sum(g['credit']['zero_variance'] for g in groups[:len(transitions)]),
              'episodes_replayed': sum(len(g['episodes']) for g in groups),
              'actions_replayed': sum(g['actions'] for g in groups),
              'generated_response_tokens': sum(g['response_tokens'] for g in groups),
              'probability_probe_extra_tokens': sum(g['probability_probe']['extra_generated_tokens'] for g in groups),
              'maximum_context_tokens': max((g['max_context_tokens'] for g in groups), default=0),
              'peak_update_cuda_bytes': max((r['metrics'].get('peak_cuda_allocated_bytes', 0) for r in transitions), default=0),
              'sampling_elapsed_seconds': sum(g['elapsed_seconds'] for g in groups),
              'update_elapsed_seconds': sum(r['metrics'].get('elapsed_seconds', 0) for r in transitions),
              'final_policy': state, 'changed_tensor_count_summed': changed_tensors,
              'state_pt_returned': False, 'ancestry': {'sft': 513, 'dpo': 11, 'rl': state['updates']},
              'evaluation_performance': 'deferred_until_profile_and_formal_run'}
    if groups and refreshed:
        result['effective_bf16_tensor_hash_changed'] = (
            groups[0]['worker_proof'][0]['loaded_tensor_sha256'] != refreshed[-1]['worker_proof'][0]['loaded_tensor_sha256'])
    atomic_json(output/('local_audit.json' if local else 'audit.json'), result)
    if not result['passed']:
        raise ValueError('preflight acceptance checks failed: '+str(checks))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--local', action='store_true')
    args = p.parse_args()
    audit(args.output, args.local)
