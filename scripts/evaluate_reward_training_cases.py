"""CPU matched greedy/probability evaluation of every actual reward-comparison training case."""

import argparse
from collections import Counter
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from collect_vllm_rollout import read, write
from diagnose_grpo_effectiveness import jsonl, write_rows, canonical_tokens, score_completion
from graded_arithmetic_reward import grade_answer
from train_grpo_first import predict
from agentic_post_training.data.preferences import load_preference_splits
from agentic_post_training.trainers.policy_iteration import file_hash, verify_files
from agentic_post_training.trainers.dpo import parameter_digest

PLAN = ROOT / 'configs/experiments/grpo_training_case_evaluation.json'
OUTPUT = ROOT / 'artifacts/evaluations/grpo-training-case-evaluation-20261028-02'
MODELS = ('parent', 'equivalent_binary', 'graded')



def predict_case_batch(model, tok, cases):
    """Left-pad one matched inference batch; retain real EOS and remove only post-EOS padding."""
    import torch
    from agentic_post_training.rewards.contract import score_reward
    prompts = [tok.apply_chat_template([{'role': 'system', 'content': c.system},
        {'role': 'user', 'content': c.prompt}], tokenize=True, add_generation_prompt=True) for c in cases]
    width = max(map(len, prompts))
    ids = torch.tensor([[tok.pad_token_id] * (width-len(p)) + p for p in prompts], dtype=torch.long)
    mask = torch.tensor([[0] * (width-len(p)) + [1] * len(p) for p in prompts], dtype=torch.long)
    with torch.inference_mode():
        generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, num_beams=1,
            use_cache=True, max_new_tokens=64, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if generated.shape[0] != len(cases) or generated.shape[1] <= width or not torch.equal(generated[:, :width], ids):
        raise ValueError('generation batch count or preserved prompt differs')
    rows = []
    for case, output in zip(cases, generated):
        tokens = output[width:].tolist()
        if tok.eos_token_id in tokens:
            end = tokens.index(tok.eos_token_id) + 1
            if any(t != tok.pad_token_id for t in tokens[end:]):
                raise ValueError('non-padding tokens after generated EOS')
            tokens, finish = tokens[:end], 'eos'
        else:
            if len(tokens) != 64:
                raise ValueError('non-EOS answer must reach the fixed length cap')
            finish = 'length'
        text = tok.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        rows.append({'case_id': case.id, 'category': case.category, 'response_token_ids': tokens,
            'finish_reason': finish, 'text': text, **score_reward(text, case, finish_reason=finish)})
    return rows


def validate_coverage(rows, case_ids):
    expected = [(model, key) for model in MODELS for key in case_ids]
    if [(r['model'], r['case_id']) for r in rows] != expected or len(set(expected)) != 72:
        raise ValueError('all three models and all 24 cases exactly once in frozen order required')


def summarize(rows, case_ids):
    validate_coverage(rows, case_ids)
    result = {}
    def totals(part):
        return {'cases': len(part),
            'strict': sum(r['reward'] == 1. for r in part),
            'verified_correct': sum(r['verified_equivalent_reward'] == 1. for r in part),
            'graded_mean': sum(r['graded_reward'] for r in part) / len(part),
            'semantic_completed': sum(r['semantic_pass'] and r['finish_reason'] == 'eos' for r in part),
            'invalid_json': sum(not r['score']['json_valid'] for r in part),
            'length': sum(r['finish_reason'] == 'length' for r in part)}
    for model in MODELS:
        part = [r for r in rows if r['model'] == model]
        result[model] = {'overall': totals(part),
            'by_role': {role: totals([r for r in part if r['role'] == role]) for role in sorted({r['role'] for r in part})},
            'by_stratum': {name: totals([r for r in part if r['wording'] + '/' + r['category'] == name])
                for name in sorted({r['wording'] + '/' + r['category'] for r in part})}}
    return result


def compare(rows):
    parent = {r['case_id']: r for r in rows if r['model'] == 'parent'}
    result = {}
    for model in MODELS[1:]:
        after = [r for r in rows if r['model'] == model]
        changes = []
        for row in after:
            before = parent[row['case_id']]
            if row['response_token_ids'] != before['response_token_ids']:
                changes.append({'case_id': row['case_id'], 'role': row['role'],
                    'before': before['text'], 'after': row['text'],
                    'before_strict': before['reward'] == 1., 'after_strict': row['reward'] == 1.,
                    'before_verified': before['verified_equivalent_reward'] == 1.,
                    'after_verified': row['verified_equivalent_reward'] == 1.})
        result[model] = {'changed_cases': changes,
            'verified_improved_case_ids': [r['case_id'] for r in after
                if parent[r['case_id']]['verified_equivalent_reward'] == 0 and r['verified_equivalent_reward'] == 1],
            'verified_regressed_case_ids': [r['case_id'] for r in after
                if parent[r['case_id']]['verified_equivalent_reward'] == 1 and r['verified_equivalent_reward'] == 0],
            'strict_improved_case_ids': [r['case_id'] for r in after
                if parent[r['case_id']]['reward'] == 0 and r['reward'] == 1],
            'strict_regressed_case_ids': [r['case_id'] for r in after
                if parent[r['case_id']]['reward'] == 1 and r['reward'] == 0]}
    return result


def probability_summary(probes):
    result = {}
    for model in MODELS[1:]:
        result[model] = {}
        for role in sorted({r['role'] for r in probes} | {'all'}):
            part = probes if role == 'all' else [r for r in probes if r['role'] == role]
            paired = [r for r in part if 'wrong' in r['models']['parent']]
            deltas = [r['models'][model]['canonical']['sequence_logp'] -
                      r['models']['parent']['canonical']['sequence_logp'] for r in part]
            margins, gaps = [], []
            for row in paired:
                a, b = row['models']['parent'], row['models'][model]
                margins.append((b['canonical']['sequence_logp'] - b['wrong']['sequence_logp']) -
                               (a['canonical']['sequence_logp'] - a['wrong']['sequence_logp']))
                gaps.append(b['canonical']['first_difference']['gap_correct_minus_old_wrong'] -
                            a['canonical']['first_difference']['gap_correct_minus_old_wrong'])
            result[model][role] = {'cases': len(part), 'canonical_logp_up': sum(x > 1e-6 for x in deltas),
                'canonical_logp_down': sum(x < -1e-6 for x in deltas),
                'mean_canonical_logp_delta': sum(deltas)/len(deltas),
                'parent_verified_wrong_cases': len(paired),
                'correct_wrong_margin_up': sum(x > 1e-6 for x in margins),
                'correct_wrong_margin_down': sum(x < -1e-6 for x in margins),
                'mean_correct_wrong_margin_delta': sum(margins)/len(margins) if margins else None,
                'first_difference_gap_up': sum(x > 1e-6 for x in gaps),
                'first_difference_gap_down': sum(x < -1e-6 for x in gaps),
                'mean_first_difference_gap_delta': sum(gaps)/len(gaps) if gaps else None,
                'correct_token_first_at_parent': sum(r['models']['parent']['canonical']['first_difference']['correct_token_rank'] == 1 for r in paired),
                'correct_token_first_after': sum(r['models'][model]['canonical']['first_difference']['correct_token_rank'] == 1 for r in paired)}
    return result


def context():
    from transformers import AutoTokenizer
    from freeze_sft_baseline import verify
    plan = read(PLAN)
    fixed = {'contract_version': 'grpo-training-case-evaluation-v2', 'device': 'cpu', 'cpu_threads': 2,
        'dtype': 'float32', 'attention': 'eager', 'do_sample': False, 'num_beams': 1, 'max_new_tokens': 64,
        'use_cache': True, 'batch_size': 24, 'padding_side': 'left', 'optimizer_updates': 0, 'sampling_temperature_experiment': False}
    if any(plan.get(k) != v for k, v in fixed.items()):
        raise ValueError('fixed CPU evaluation contract differs')
    verify_files(ROOT, plan['input_sha256'])
    comparison = ROOT / 'artifacts/checkpoints/reward-comparison-20261028-01'
    for name in ('shared', 'equivalent_binary', 'graded'):
        folder = comparison / name
        summary = read(folder / 'summary.json')
        verify_files(folder, summary['files_sha256'])
        sources = read(folder / 'run_manifest.json')['source_sha256']
        verify_files(folder / 'source', sources)
        verify_files(ROOT, sources)
    original_plan = read(ROOT / 'configs/experiments/grpo_reward_comparison.json')
    if plan['case_ids'] != original_plan['case_ids'] or len(plan['case_ids']) != 24:
        raise ValueError('all actual training cases required')
    baseline = ROOT / 'artifacts/baselines/sft-v2-phase3'
    verify(baseline)
    base_manifest = read(baseline / 'manifest.json')
    base = (ROOT / base_manifest['base_cache_files'][0]['source']).parent
    folder = ROOT / 'data/preferences/dpo_v2'
    splits = load_preference_splits(folder, read(folder / 'manifest.json'))
    indexed = {r.id: r for r in splits['train']}
    selected = [indexed[k] for k in plan['case_ids']]
    if ({r.id for r in selected} & {r.id for r in splits['development']}
            or {r.family_id for r in selected} & {r.family_id for r in splits['development']}):
        raise ValueError('development input entered training-case evaluation')
    tok = AutoTokenizer.from_pretrained(str(baseline / 'checkpoint/tokenizer'), local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prompts, canonical = [], []
    for row in selected:
        prompt, answer = canonical_tokens(row, tok)
        prompts.append(prompt)
        canonical.append(answer)
    anchors = {model: {r['case_id']: r for r in jsonl(ROOT / plan['anchors'][model])} for model in MODELS}
    return plan, base, tok, selected, prompts, canonical, anchors


def run(output):
    import torch
    from peft import PeftModel, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM
    started = time.perf_counter()
    plan, base, tok, selected, prompts, canonical, anchors = context()
    if output.exists():
        raise FileExistsError('fresh evaluation output required; no automatic retry')
    output.mkdir(parents=True, exist_ok=False)
    sources = read(ROOT / 'artifacts/checkpoints/reward-comparison-20261028-01/shared/run_manifest.json')['source_sha256'].copy()
    for path in (Path(__file__), PLAN):
        sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    manifest = {'plan': plan, 'source_sha256': sources, 'new_optimizer_updates': 0,
        'canonical_note': 'one strict-valid reference, not total probability mass of every equivalent answer'}
    write(output / 'run_manifest.json', manifest)
    for name, digest in sources.items():
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_hash(target) != digest:
            raise ValueError('source changed during snapshot')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(202610290)
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation='eager')
    model = PeftModel.from_pretrained(model, str(ROOT / plan['adapters']['parent']), is_trainable=False,
        autocast_adapter_dtype=False).eval().requires_grad_(False)
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in model.parameters()) or any(isinstance(m, torch.nn.Dropout) and m.p != 0 for m in model.modules()):
        raise ValueError('dropout-free FP32 CPU model required')
    rows, parent_rows, checks = [], {}, {}
    probes = [{'case_id': r.id, 'role': ('persistent_zero' if i % 6 < 4 else 'near_control' if i % 6 == 4 else 'easy_control'),
               'models': {}} for i, r in enumerate(selected)]
    counter = {'forwards': 0}
    def count_forward(module, args):
        counter['forwards'] += 1
        if counter['forwards'] % 8 == 0:
            (output / 'progress.json').write_text(json.dumps({'model': label, 'forward_calls': counter['forwards'], 'elapsed_seconds': time.perf_counter()-started}) + '\n', encoding='utf-8')
    handle = model.get_base_model().register_forward_pre_hook(count_forward)
    for label in MODELS:
        before_calls = counter['forwards']
        if label != 'parent':
            loaded = set_peft_model_state_dict(model, load_file(str(ROOT / plan['adapters'][label] / 'adapter_model.safetensors')))
            if loaded.unexpected_keys or any('lora_' in key for key in loaded.missing_keys):
                raise ValueError('adapter switch incompletely loaded')
        initial = parameter_digest(model)
        if initial != plan['parameter_sha256'][label] or any(p.requires_grad for p in model.parameters()):
            raise ValueError('frozen model checkpoint identity differs')
        model_rows = []
        batch_predictions = predict_case_batch(model, tok, [example.target() for example in selected])
        print(f'CPU batch decoded {label} 24/24', flush=True)
        with (output / (label + '_predictions.jsonl')).open('x', encoding='utf-8', newline='\n') as stream:
            for i, (example, prediction) in enumerate(zip(selected, batch_predictions)):
                row = {'model': label, 'wording': example.provenance['wording'], 'family_id': example.family_id,
                    'role': probes[i]['role'], 'case': asdict(example.target()), **prediction,
                    **grade_answer(prediction['text'], example.target(), finish_reason=prediction['finish_reason'])}
                row['anchor_available'] = example.id in anchors[label]
                row['anchor_native_token_match'] = (row['response_token_ids'] == anchors[label][example.id]['response_token_ids']) if row['anchor_available'] else None
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                model_rows.append(row)
                if label == 'parent':
                    parent_rows[example.id] = row
                print(f'CPU greedy {label} {i+1}/24', flush=True)
        generation_calls = counter['forwards'] - before_calls
        teacher_calls = 0
        for i, example in enumerate(selected):
            old = parent_rows[example.id]
            wrong = old['response_token_ids'] if old['verified_equivalent_reward'] == 0. else None
            values = {'canonical': score_completion(model, prompts[i], canonical[i], wrong)}
            teacher_calls += 1
            if wrong is not None:
                values['wrong'] = score_completion(model, prompts[i], wrong)
                teacher_calls += 1
            probes[i]['models'][label] = values
        anchor_rows = [r for r in model_rows if r['anchor_available']]
        checks[label] = {'expected_parameter_sha256': initial, 'final_parameter_sha256': parameter_digest(model),
            'parameters_unchanged': parameter_digest(model) == initial, 'all_frozen': all(not p.requires_grad for p in model.parameters()),
            'gradients_absent': all(p.grad is None for p in model.parameters()), 'greedy_sequences': 24,
            'generation_forward_calls': generation_calls, 'teacher_forced_forward_calls': teacher_calls,
            'anchor_count': len(anchor_rows), 'anchor_native_matches': sum(r['anchor_native_token_match'] for r in anchor_rows),
            'anchor_mismatched_case_ids': [r['case_id'] for r in anchor_rows if not r['anchor_native_token_match']]}
        if not checks[label]['parameters_unchanged'] or not checks[label]['gradients_absent']:
            raise ValueError('evaluation changed parameters or accumulated gradients')
        write(output / (label + '_checks.json'), checks[label])
        print(f'CPU model {label} completed; anchors {checks[label]["anchor_native_matches"]}/{checks[label]["anchor_count"]}', flush=True)
    handle.remove()
    write_rows(output / 'probability_scores.jsonl', probes)
    verify_files(ROOT, plan['input_sha256'])
    verify_files(ROOT, sources)
    result = {'contract_version': plan['contract_version'], 'device': 'cpu', 'cpu_threads': 2,
        'evaluation_cases': 24, 'greedy_sequences': 72, 'evaluation': summarize(rows, plan['case_ids']),
        'comparison_to_parent': compare(rows), 'probability_changes': probability_summary(probes),
        'model_checks': checks, 'total_model_forward_calls': counter['forwards'],
        'all_21_cloud_overlap_anchors_match': all(c['anchor_count'] == c['anchor_native_matches'] == 7 for c in checks.values()),
        'cross_device_limit': '21 overlap anchors only; CPU outputs are a separate uniform supplementary evaluation, not a GPU replay guarantee',
        'canonical_probability_limit': 'one reference continuation; changes do not equal changes in total verified-correct probability mass',
        'backward_calls': 0, 'optimizer_updates': 0, 'baseline_promoted': False,
        'elapsed_seconds': time.perf_counter()-started,
        'files_sha256': {p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob('*') if p.is_file()}}
    write(output / 'summary.json', result)
    print(json.dumps({k: result[k] for k in ('evaluation', 'comparison_to_parent', 'probability_changes', 'all_21_cloud_overlap_anchors_match', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output)
