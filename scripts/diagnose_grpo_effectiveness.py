"""Read existing GRPO evidence and teacher-force fixed probes on CPU; no training."""

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.evaluation.tool_calls import score_output
from agentic_post_training.rewards.audit import audit_records
from agentic_post_training.rewards.contract import score_reward
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files

OLD = ROOT / 'artifacts/trajectories/phase5-signal-20261021-01'
NEW = ROOT / 'artifacts/trajectories/grpo-refresh-20261028-01'
FIRST = ROOT / 'artifacts/checkpoints/grpo-first-20261027-01'
CURRENT = ROOT / 'artifacts/checkpoints/grpo-iteration-20261028-01'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_rows(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def strict(row):
    return bool(row['strict_pass'] and row['finish_reason'] == 'eos')


def error_kind(row, case):
    if strict(row):
        return 'passed'
    if row['finish_reason'] != 'eos':
        return 'length_or_abnormal_stop'
    score = row['score']
    for key, label in [('json_valid', 'invalid_json'), ('schema_valid', 'invalid_schema'),
                       ('route_correct', 'wrong_route'), ('arguments_valid', 'invalid_arguments'),
                       ('tool_execution_success', 'execution_failure')]:
        if case.expected_type == 'final' and key in ('arguments_valid', 'tool_execution_success'):
            continue
        if not score[key]:
            return label
    if row['semantic_pass']:
        return 'value_correct_ast_mismatch'
    if case.expected_type == 'tool_call':
        expression = json.loads(row['text'])['arguments']['expression']
        def components(text):
            tree = ast.parse(text, mode='eval')
            return (Counter(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)),
                    Counter(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, ast.BinOp)))
        got_numbers, got_ops = components(expression)
        expected_numbers, expected_ops = components(case.expected_expression)
        if got_numbers != expected_numbers:
            return 'wrong_number_multiset'
        if got_ops != expected_ops:
            return 'wrong_operator_multiset'
        return 'wrong_order_or_grouping'
    return 'other_failure'


def group_rows(rows, fields):
    buckets = defaultdict(list)
    for row in rows:
        buckets[' / '.join(str(row[key]) for key in fields)].append(row)
    return dict(sorted(buckets.items()))


def prediction_summary(rows):
    return {key: {'cases': len(values), 'strict': sum(r['strict'] for r in values),
                   'semantic': sum(r['semantic'] for r in values),
                   'errors': dict(Counter(r['error_kind'] for r in values))}
            for key, values in group_rows(rows, ('suite', 'wording')).items()}


def canonical_tokens(example, tok):
    messages = [{'role': 'system', 'content': example.system}, {'role': 'user', 'content': example.prompt}]
    prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(messages + [{'role': 'assistant', 'content': example.chosen}],
                                   tokenize=True, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError('canonical target has incompatible prompt boundary')
    response = full[len(prompt):]
    if tok.eos_token_id not in response:
        raise ValueError('canonical target must contain EOS')
    end = response.index(tok.eos_token_id) + 1
    if response[end:] and not tok.decode(response[end:], skip_special_tokens=False).isspace():
        raise ValueError('unexpected non-whitespace template suffix after EOS')
    response = response[:end]
    if len(response) > 64 or len(prompt) + len(response) > 512:
        raise ValueError('canonical probe exceeds frozen context/completion budget')
    if tok.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False) != example.chosen:
        raise ValueError('canonical target token decoding differs')
    if not score_output(example.chosen, example.target())['end_to_end_pass']:
        raise ValueError('canonical target does not satisfy strict task')
    return prompt, response


def records(output):
    from iteration_common import setup
    _, training_plan, baseline, base, tok, cases, prompts = setup()
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for folder in (OLD, NEW, FIRST, CURRENT):
        saved = read(folder / 'summary.json')
        verify_files(folder, saved['files_sha256'])
        inv = read(folder / 'run_manifest.json')
        verify_files(folder / 'source', inv['source_sha256'])
        verify_files(ROOT, inv['source_sha256'])
        for name in ('summary.json', 'run_manifest.json', *saved['files_sha256']):
            path = folder / name
            files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    folder = ROOT / 'data/preferences/dpo_v2'
    splits = load_preference_splits(folder, read(folder / 'manifest.json'))
    examples = {r.id: r for rows in splits.values() for r in rows}
    for name in ('manifest.json', *read(folder / 'manifest.json')['files']):
        path = folder / name
        files[path.relative_to(ROOT).as_posix()] = file_hash(path)
    files['artifacts/baselines/sft-v2-phase3/manifest.json'] = file_hash(baseline / 'manifest.json')
    files['scripts/diagnose_grpo_effectiveness.py'] = file_hash(Path(__file__))
    metadata = lambda example: {'case_id': example.id, 'wording': example.provenance['wording'],
                                'category': example.category, 'family_id': example.family_id}
    rollout_cases = {}
    raw_rollouts = {}
    for label, source, seed_base in [('sft', OLD, 20261021), ('previous', NEW, 202610280)]:
        raw = jsonl(source / 'rollouts.jsonl')
        audit = audit_records(raw)
        if audit != read(source / 'reward_audit.json'):
            raise ValueError('saved reward audit differs')
        for g, case in enumerate(cases):
            for i, row in enumerate(raw[g * 8:(g + 1) * 8]):
                sample = row['sample']
                if (row['case'] != asdict(case) or sample['seed'] != seed_base + g * 1000 + i
                        or sample['prompt_token_ids'] != prompts[g]
                        or sample['text'] != tok.decode(sample['response_token_ids'], skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)):
                    raise ValueError('native rollout identity/decode differs')
        indexed = {g['group_id']: g for g in audit['groups']}
        if len(raw) != 1152 or set(indexed) != set(training_plan['case_ids']):
            raise ValueError('all 144x8 training records required')
        sample_buckets = defaultdict(list)
        for row in raw:
            sample_buckets[row['sample']['case_id']].append(row)
        raw_rollouts[label] = sample_buckets
        rollout_cases[label] = []
        for case in cases:
            group = indexed[case.id]
            values = group['rewards']
            rows = sample_buckets[case.id]
            errors = Counter(error_kind({'finish_reason': r['sample']['finish_reason'],
                                         'strict_pass': r['score']['end_to_end_pass'],
                                         'semantic_pass': r['semantic_pass'], 'score': r['score'],
                                         'text': r['sample']['text']}, case) for r in rows)
            rollout_cases[label].append({**metadata(examples[case.id]), 'samples': 8,
                'strict': int(sum(values)), 'semantic': sum(r['semantic_pass'] for r in rows),
                'mixed': 0 < sum(values) < 8, 'errors': dict(errors),
                'unique_native_answers': len({tuple(r['sample']['response_token_ids']) for r in rows})})
    predictions = {}
    pred_maps = {}
    for label, filename in [('sft', 'sft_predictions.jsonl'), ('previous', 'previous_predictions.jsonl'),
                             ('current', 'current_predictions.jsonl')]:
        rows = jsonl(CURRENT / filename)
        pred_maps[label] = {r['case_id']: r for r in rows}
        if len(rows) != 104 or len(pred_maps[label]) != 104:
            raise ValueError('104 unique predictions required per checkpoint')
        derived = []
        for row in rows:
            example = examples[row['case_id']]
            case = example.target()
            rw = score_reward(row['text'], case, finish_reason=row['finish_reason'])
            if any(row[k] != v for k, v in rw.items()):
                raise ValueError('prediction rescoring mismatch')
            if tok.decode(row['response_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False) != row['text']:
                raise ValueError('prediction native decode differs')
            derived.append({**metadata(example), 'suite': row['suite'], 'strict': strict(row),
                            'semantic': row['semantic_pass'], 'error_kind': error_kind(row, case)})
        predictions[label] = derived
    old_groups = {r['case_id']: r for r in rollout_cases['sft']}
    new_groups = {r['case_id']: r for r in rollout_cases['previous']}
    case_diagnostics = []
    for identifier in training_plan['case_ids'] + training_plan['evaluation_development_case_ids']:
        example = examples[identifier]
        row = {**metadata(example), 'split': 'train' if identifier in old_groups else 'development',
               'prompt': example.prompt, 'expected_expression': example.expected_expression,
               'rollout_sft': old_groups.get(identifier), 'rollout_previous': new_groups.get(identifier),
               'greedy': {label: next((r for r in values if r['case_id'] == identifier), None)
                          for label, values in predictions.items()}}
        case_diagnostics.append(row)
    wrong_train = [r for r in case_diagnostics if r['split'] == 'train'
                   and r['greedy']['previous'] and not r['greedy']['previous']['strict']]
    coverage = {key: {'wrong_greedy_cases': len(rows),
                      'no_correct_in_eight': sum(r['rollout_previous']['strict'] == 0 for r in rows),
                      'mixed': sum(r['rollout_previous']['mixed'] for r in rows),
                      'all_correct': sum(r['rollout_previous']['strict'] == 8 for r in rows),
                      'now_greedy_correct': sum(r['greedy']['current']['strict'] for r in rows)}
                for key, rows in group_rows(wrong_train, ('wording',)).items()}
    probes = []
    for row in jsonl(CURRENT / 'previous_predictions.jsonl'):
        example = examples[row['case_id']]
        if example.expected_type != 'tool_call':
            continue
        prompt, target = canonical_tokens(example, tok)
        probe = {**metadata(example), 'suite': row['suite'], 'prompt_token_ids': prompt,
                 'canonical_token_ids': target, 'canonical_text': example.chosen,
                 'previous_greedy_strict': strict(row), 'current_greedy_strict': strict(pred_maps['current'][example.id]),
                 'previous_greedy_token_ids': row['response_token_ids'], 'previous_greedy_text': row['text'],
                 'new_rollout_correct_count': new_groups[example.id]['strict'] if example.id in new_groups else None}
        if not strict(row) and example.id in new_groups and new_groups[example.id]['strict']:
            positive = [r['sample'] for r in raw_rollouts['previous'][example.id]
                        if score_reward(r['sample']['text'], example.target(), finish_reason=r['sample']['finish_reason'])['reward'] == 1]
            positive.sort(key=lambda s: s['sample_index'])
            if positive[0]['response_token_ids'] != target:
                probe['sampled_correct_token_ids'] = positive[0]['response_token_ids']
                probe['sampled_correct_text'] = positive[0]['text']
        probes.append(probe)
    if len(probes) != 96:
        raise ValueError('exactly 48 train and 48 development arithmetic probes required')
    transitions = Counter(('zero' if old_groups[k]['strict'] == 0 else 'all' if old_groups[k]['strict'] == 8 else 'mixed')
                          + ' -> ' + ('zero' if new_groups[k]['strict'] == 0 else 'all' if new_groups[k]['strict'] == 8 else 'mixed')
                          for k in old_groups)
    rollout_summary = {}
    for label, rows in rollout_cases.items():
        rollout_summary[label] = {key: {'groups': len(values), 'samples': 8 * len(values),
            'strict': sum(v['strict'] for v in values), 'semantic': sum(v['semantic'] for v in values),
            'zero_correct_groups': sum(v['strict'] == 0 for v in values),
            'all_correct_groups': sum(v['strict'] == 8 for v in values), 'mixed_groups': sum(v['mixed'] for v in values),
            'errors': dict(sum((Counter(v['errors']) for v in values), Counter()))}
            for key, values in group_rows(rows, ('wording',)).items()}
    result = {'prediction_by_wording': {label: prediction_summary(rows) for label, rows in predictions.items()},
              'prediction_by_structure': {label: {key: {'cases': len(values), 'strict': sum(v['strict'] for v in values),
                    'semantic': sum(v['semantic'] for v in values), 'errors': dict(Counter(v['error_kind'] for v in values))}
                    for key, values in group_rows(rows, ('suite', 'wording', 'category')).items()}
                    for label, rows in predictions.items()},
              'rollout_by_wording': rollout_summary, 'rollout_outcome_transitions': dict(transitions),
              'wrong_train_greedy_sampling_coverage': coverage,
              'both_rollouts_no_correct_case_ids': [k for k in old_groups if old_groups[k]['strict'] == new_groups[k]['strict'] == 0],
              'rollout_histograms': {label: dict(Counter(str(r['strict']) for r in rows)) for label, rows in rollout_cases.items()},
              'new_training_updates': 0, 'new_generation_calls': 0, 'new_model_forward_calls': 0}
    write_rows(output / 'case_diagnostics.jsonl', case_diagnostics)
    write_rows(output / 'probability_probes.jsonl', probes)
    write(output / 'record_diagnostic.json', result)
    plan = {'contract_version': 'grpo-effectiveness-diagnostic-v1', 'input_files_sha256': files,
            'probe_file_sha256': file_hash(output / 'probability_probes.jsonl'), 'probe_cases': len(probes),
            'selection': 'all 48 existing train diagnostic and 48 existing development arithmetic cases; no new generations',
            'continuations': 'canonical chosen; old incorrect greedy native IDs; first new-rollout strict-positive native answer if distinct',
            'scoring': 'CPU FP32 eager, no padding, 2 threads, no gradients, no weight updates',
            'canonical_note': 'one strict-valid reference string, not the total probability mass of every accepted answer',
            'models': {'sft': 'artifacts/baselines/sft-v2-phase3/checkpoint/adapter',
                       'previous': 'artifacts/checkpoints/grpo-first-20261027-01/final-checkpoint/adapter',
                       'current': 'artifacts/checkpoints/grpo-iteration-20261028-01/final-checkpoint/adapter'},
            'parameter_sha256': {'sft': read(ROOT / 'configs/trainer_probability_contract.json')['frozen_fp32_parameter_sha256'],
                                  'previous': training_plan['parent_parameter_sha256'],
                                  'current': read(CURRENT / 'summary.json')['final_parameter_sha256']},
            'model_weight_files_sha256': {entry['source']: entry['sha256'] for entry in read(baseline / 'manifest.json')['base_cache_files']}}
    verify_files(ROOT, plan['input_files_sha256'])
    write(output / 'diagnostic_plan.json', plan)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def first_difference(left, right):
    if left == right:
        return None
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    raise ValueError('one continuation is a strict prefix; complete EOS-terminated probes required')


def score_completion(model, prompt, response, contrast=None):
    import torch
    from agentic_post_training.algorithms.losses import causal_token_log_probs
    if not prompt or not response or model.training:
        raise ValueError('eval-mode model and nonempty prompt/response required')
    with torch.inference_mode():
        ids = torch.tensor([prompt + response], dtype=torch.long)
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits.float()
        values = causal_token_log_probs(logits, ids)[0, len(prompt) - 1:]
        if not torch.isfinite(values).all():
            raise ValueError('nonfinite diagnostic logp')
        result = {'tokens': len(response), 'sequence_logp': values.double().sum().item(),
                  'mean_token_logp': values.double().mean().item(), 'token_logps': values.tolist()}
        if contrast is not None:
            index = first_difference(response, contrast)
            if index is None:
                raise ValueError('correct and incorrect continuations must differ')
            distribution = logits[0, len(prompt) - 1 + index].log_softmax(-1)
            correct, wrong = response[index], contrast[index]
            result['first_difference'] = {'response_index': index, 'correct_token_id': correct,
                'old_wrong_token_id': wrong, 'correct_logp': distribution[correct].item(),
                'old_wrong_logp': distribution[wrong].item(),
                'gap_correct_minus_old_wrong': (distribution[correct] - distribution[wrong]).item(),
                'correct_token_rank': int((distribution > distribution[correct]).sum()) + 1,
                'top_token_id': int(distribution.argmax())}
        return result


def probability_changes(rows, before, after):
    def summarize(values):
        delta = [r['models'][after]['canonical']['sequence_logp'] - r['models'][before]['canonical']['sequence_logp'] for r in values]
        paired = [r for r in values if 'wrong' in r['models'][before]]
        margins = [(r['models'][after]['canonical']['sequence_logp'] - r['models'][after]['wrong']['sequence_logp'])
                   - (r['models'][before]['canonical']['sequence_logp'] - r['models'][before]['wrong']['sequence_logp']) for r in paired]
        gaps = [r['models'][after]['canonical']['first_difference']['gap_correct_minus_old_wrong']
                - r['models'][before]['canonical']['first_difference']['gap_correct_minus_old_wrong'] for r in paired]
        ranks = lambda label: Counter(str(r['models'][label]['canonical']['first_difference']['correct_token_rank']) for r in paired)
        return {'cases': len(values), 'mean_canonical_sequence_logp_delta': sum(delta) / len(delta),
                'canonical_logp_up': sum(v > 1e-6 for v in delta), 'canonical_logp_down': sum(v < -1e-6 for v in delta),
                'old_wrong_cases': len(paired), 'mean_correct_wrong_sequence_margin_delta': sum(margins) / len(margins) if margins else None,
                'sequence_margin_up': sum(v > 1e-6 for v in margins),
                'mean_first_difference_gap_delta': sum(gaps) / len(gaps) if gaps else None,
                'first_difference_gap_up': sum(v > 1e-6 for v in gaps),
                'first_difference_correct_rank_before': dict(ranks(before)),
                'first_difference_correct_rank_after': dict(ranks(after))}
    groups = group_rows(rows, ('suite', 'wording'))
    groups.update({suite + ' / all': values for suite, values in group_rows(rows, ('suite',)).items()})
    return {key: summarize(values) for key, values in groups.items()}


def probabilities(output):
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    from agentic_post_training.trainers.dpo import parameter_digest
    started = time.perf_counter()
    plan = read(output / 'diagnostic_plan.json')
    verify_files(ROOT, plan['input_files_sha256'])
    verify_files(ROOT, plan['model_weight_files_sha256'])
    if file_hash(output / 'probability_probes.jsonl') != plan['probe_file_sha256']:
        raise ValueError('diagnostic probes changed')
    if (output / 'probability_diagnostic.json').exists() or (output / 'probability_scores.jsonl').exists():
        raise FileExistsError('probability outputs already exist')
    probes = jsonl(output / 'probability_probes.jsonl')
    torch.set_num_threads(2)
    torch.manual_seed(20261028)
    torch.set_float32_matmul_precision('highest')
    base = (ROOT / next(iter(plan['model_weight_files_sha256']))).parent
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager')
    model = PeftModel.from_pretrained(model, str(ROOT / plan['models']['sft']), is_trainable=False,
                                     autocast_adapter_dtype=False).eval()
    forward_calls = 0
    results = [{**probe, 'models': {}} for probe in probes]
    checkpoint_checks = {}
    for label, adapter_path in plan['models'].items():
        adapter = load_file(str(ROOT / adapter_path / 'adapter_model.safetensors'))
        loaded = set_peft_model_state_dict(model, adapter)
        if loaded.unexpected_keys or any('lora_' in key for key in loaded.missing_keys):
            raise ValueError('incomplete diagnostic adapter load')
        before = parameter_digest(model)
        if before != plan['parameter_sha256'][label]:
            raise ValueError('diagnostic model identity differs')
        for i, row in enumerate(results, 1):
            contrast = row['previous_greedy_token_ids'] if not row['previous_greedy_strict'] else None
            scored = {'canonical': score_completion(model, row['prompt_token_ids'], row['canonical_token_ids'], contrast)}
            forward_calls += 1
            if contrast is not None:
                scored['wrong'] = score_completion(model, row['prompt_token_ids'], contrast)
                forward_calls += 1
            if 'sampled_correct_token_ids' in row:
                scored['sampled_correct'] = score_completion(model, row['prompt_token_ids'], row['sampled_correct_token_ids'])
                forward_calls += 1
            row['models'][label] = scored
            if i % 16 == 0:
                print(f'CPU fixed-probe scoring {label}: {i}/{len(results)}; elapsed {time.perf_counter()-started:.1f}s', flush=True)
        after = parameter_digest(model)
        if after != before or any(p.grad is not None for p in model.parameters()):
            raise ValueError('diagnostic unexpectedly changed weights or accumulated gradients')
        checkpoint_checks[label] = {'parameter_sha256': before, 'parameters_unchanged': True, 'gradients_absent': True}
    verify_files(ROOT, plan['input_files_sha256'])
    write_rows(output / 'probability_scores.jsonl', results)
    result = {'contract_version': plan['contract_version'], 'elapsed_seconds': time.perf_counter() - started,
              'cpu_threads': 2, 'device': 'cpu', 'dtype': 'float32', 'attention': 'eager',
              'probe_cases': len(probes), 'model_forward_calls': forward_calls, 'new_generation_calls': 0,
              'backward_calls': 0, 'optimizer_updates': 0, 'checkpoint_checks': checkpoint_checks,
              'sft_to_previous': probability_changes(results, 'sft', 'previous'),
              'previous_to_current': probability_changes(results, 'previous', 'current'),
              'sft_to_current': probability_changes(results, 'sft', 'current'),
              'notes': ['canonical score concerns one valid answer, not total valid-answer probability mass',
                        'sequence logp comparison is length-sensitive; first-difference gap shares identical prefix',
                        'first-difference preference does not guarantee whole-sequence greedy correctness',
                        'CPU scores are diagnostic, not a replay of the GPU training numerical gate'],
              'scores_sha256': file_hash(output / 'probability_scores.jsonl')}
    write(output / 'probability_diagnostic.json', result)
    print(json.dumps({k: result[k] for k in ('elapsed_seconds', 'probe_cases', 'model_forward_calls', 'new_generation_calls', 'optimizer_updates')}), flush=True)


def main():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('records', 'probabilities'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/evaluations/grpo-effectiveness-diagnostic-20261028-01')
    args = parser.parse_args()
    if args.stage == 'records':
        records(args.output)
    else:
        probabilities(args.output)


if __name__ == '__main__':
    main()
