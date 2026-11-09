"""Formal training/evaluation audit, complete paired tables and descriptive ranking."""
import argparse
import csv
import sys
from pathlib import Path
from grpo_common import (read, atomic_json, file_hash, verify_inputs, initial_policy, load_transition)
from apt_v2.grpo_evaluation import paired_evaluation
from evaluate_v2_grpo import verify_evaluation


def audit_formal(output, local=False):
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm
    from audit_v2_grpo import audit as audit_training
    cfg = verify_inputs(output, runtime=not local, local=local)
    if cfg['phase'] != 'formal':
        raise ValueError('formal configuration required')
    train = audit_training(output, local=local)
    tok = AutoTokenizer.from_pretrained(output/'tokenizer', local_files_only=True, trust_remote_code=False)
    states = {0: initial_policy(cfg)}
    transitions = []
    for i in range(cfg['max_groups']):
        transition = load_transition(output, i, states[i])
        transitions.append(transition)
        states[i+1] = transition['after']
    branches = [('weighted_sft', 0), *[(f'grpo_g{n:04}', n) for n in cfg['group_evaluation_cursors']]]
    evaluations, audits, summaries = {}, {}, {}
    for label, cursor in tqdm(branches, desc='Paired evaluation audit', file=sys.stdout, disable=False):
        spec = read(output/'evaluations'/(label+'-policy.json'))
        if spec != {'label': label, 'group_cursor': cursor, 'policy': states[cursor]}:
            raise ValueError('evaluation checkpoint cursor differs')
        audit = verify_evaluation(output, label, spec, tok, local=local)
        if not audit['passed']:
            raise ValueError('quality/calendar evaluation audit failed')
        data = read(output/'evaluations'/(label+'.json'))
        evaluations[label] = data['episodes']
        audits[label] = {k: v for k, v in audit.items() if k != 'calendar'}
        summaries[label] = audit['summary']
    paired = {label: paired_evaluation(evaluations['weighted_sft'], episodes)
              for label, episodes in evaluations.items() if label != 'weighted_sft'}
    base = summaries['weighted_sft']['sample']['complete']
    primary = 'recovery' if base['successes'] == base['episodes'] else 'complete'
    secondary = 'complete' if primary == 'recovery' else 'recovery'
    def rank(label):
        s = summaries[label]
        return (s['sample'][primary]['successes'], s['sample'][secondary]['successes'],
                s['greedy']['complete']['successes'], s['greedy']['recovery']['successes'],
                -int(label[-4:]))
    ranked = sorted(paired, key=rank, reverse=True)
    comparison = {'branches': summaries, 'paired_to_weighted_sft': paired,
                  'primary_descriptive_metric': primary, 'candidate_ranking': ranked,
                  'performance_acceptance_thresholds': cfg['acceptance_thresholds'],
                  'selection_scope': 'development_exploration; final confirmation scheduled in C4'}
    name = 'local_comparison_summary.json' if local else 'comparison_summary.json'
    atomic_json(output/name, comparison)
    rows = []
    for tr in transitions:
        g = read(output/'groups'/f'group-{tr["index"]:04}.json')
        m = tr['metrics']
        rows.append({'group': tr['index'], 'kind': g['spec']['kind'], 'task': g['spec']['task_id'],
                     'scenario': g['spec']['scenario'], 'successes': sum(e['success'] for e in g['episodes']),
                     'rollouts': len(g['episodes']), 'reward_mean': g['credit']['mean'],
                     'reward_std': g['credit']['population_std'], 'eligible': g['credit']['eligible'],
                     'updated': tr['updated'], 'actual_updates': tr['after']['updates'],
                     'actions': g['actions'], 'response_tokens': g['response_tokens'],
                     'sampled_k3': m.get('sampled_k3', 0), 'clip_fraction': m.get('clip_fraction', 0),
                     'tis_mean': m.get('tis_mean', 0), 'grad_norm': m.get('grad_norm', 0)})
    if rows:
        with (output/('local_training_curve.csv' if local else 'training_curve.csv')).open('w', newline='', encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    curve = []
    for label, cursor in branches:
        for mode in ('greedy', 'sample'):
            for kind in ('complete', 'recovery'):
                s = summaries[label][mode][kind]
                curve.append({'branch': label, 'group_cursor': cursor, 'actual_updates': states[cursor]['updates'],
                              'mode': mode, 'kind': kind, 'successes': s['successes'], 'episodes': s['episodes'],
                              'response_tokens': s['response_tokens'], 'actions': s['actions']})
    with (output/('local_evaluation_curve.csv' if local else 'evaluation_curve.csv')).open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    result = {'passed': train['passed'] and all(a['passed'] for a in audits.values()),
              'scope': 'local_returned_files' if local else 'cloud_files_and_state',
              'training': train, 'evaluation': audits, 'candidate_ranking': ranked,
              'performance_acceptance_thresholds': cfg['acceptance_thresholds'],
              'formal_job_index': cfg['formal_job_index'], 'max_wall_seconds': cfg['max_wall_seconds'],
              'source_sft_updates': 513, 'source_dpo_updates': 11, 'actual_rl_updates': train['actual_updates']}
    atomic_json(output/('local_formal_audit.json' if local else 'formal_audit.json'), result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--local', action='store_true')
    args = p.parse_args()
    audit_formal(args.output, args.local)
