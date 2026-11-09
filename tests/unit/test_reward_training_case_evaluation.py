import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from evaluate_reward_training_cases import MODELS, validate_coverage, summarize, compare, probability_summary

IDS = [f'case-{i}' for i in range(24)]


def predictions():
    return [{'model': model, 'case_id': key, 'reward': 1., 'verified_equivalent_reward': 1.,
             'graded_reward': 1., 'semantic_pass': True, 'finish_reason': 'eos',
             'score': {'json_valid': True}, 'role': 'persistent_zero' if i < 16 else 'control',
             'wording': 'natural' if i < 8 else 'sequence', 'category': 'math',
             'response_token_ids': [4, 2], 'text': 'correct'}
            for model in MODELS for i, key in enumerate(IDS)]


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'order'])
def test_complete_case_coverage_is_required(mutation):
    rows = predictions()
    if mutation == 'missing': rows.pop()
    elif mutation == 'duplicate': rows[-1] = rows[-2]
    else: rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match='all three'):
        validate_coverage(rows, IDS)


def test_strict_and_verified_correctness_are_kept_distinct():
    rows = predictions()
    rows[0]['reward'] = 0.  # A commutatively equivalent answer may fail the original strict rule.
    result = summarize(rows, IDS)
    assert result['parent']['overall']['strict'] == 23
    assert result['parent']['overall']['verified_correct'] == 24
    assert result['parent']['by_role']['persistent_zero']['cases'] == 16


def test_truncation_does_not_count_as_completed_semantic_success():
    rows = predictions()
    rows[0].update(reward=0., verified_equivalent_reward=0., graded_reward=0., finish_reason='length')
    result = summarize(rows, IDS)['parent']['overall']
    assert result['semantic_completed'] == result['verified_correct'] == 23
    assert result['length'] == 1


def test_changed_text_and_partial_reward_are_not_new_correct_answers():
    rows = predictions()
    rows[-1].update(reward=0., verified_equivalent_reward=0., graded_reward=.3,
                    response_token_ids=[5, 2], text='wrong grouping')
    result = compare(rows)
    assert result['graded']['verified_regressed_case_ids'] == ['case-23']
    assert result['graded']['verified_improved_case_ids'] == []
    assert len(result['graded']['changed_cases']) == 1
    assert result['equivalent_binary']['changed_cases'] == []


def test_absolute_probability_rise_can_still_lose_against_the_wrong_continuation():
    before = {'canonical': {'sequence_logp': -3., 'first_difference': {
        'gap_correct_minus_old_wrong': -.5, 'correct_token_rank': 2}}, 'wrong': {'sequence_logp': -2.}}
    worse = copy.deepcopy(before)
    worse['canonical']['sequence_logp'] = -2.8
    worse['wrong']['sequence_logp'] = -1.5
    worse['canonical']['first_difference']['gap_correct_minus_old_wrong'] = -.6
    better = copy.deepcopy(before)
    better['canonical']['sequence_logp'] = -2.9
    better['wrong']['sequence_logp'] = -2.3
    better['canonical']['first_difference'].update(gap_correct_minus_old_wrong=-.4, correct_token_rank=1)
    probe = {'case_id': 'x', 'role': 'persistent_zero',
             'models': {'parent': before, 'equivalent_binary': worse, 'graded': better}}
    result = probability_summary([probe])
    left, right = result['equivalent_binary']['all'], result['graded']['all']
    assert left['canonical_logp_up'] == 1 and left['correct_wrong_margin_down'] == 1
    assert left['first_difference_gap_down'] == 1
    assert right['correct_wrong_margin_up'] == 1 and right['correct_token_first_after'] == 1


def test_batched_greedy_matches_serial_tiny_model_with_different_prompt_lengths():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from agentic_post_training.evaluation.tool_calls import EvalCase
    from agentic_post_training.trainers.dpo import parameter_digest
    from evaluate_reward_training_cases import predict_case_batch
    from train_grpo_first import predict

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return [int(x) for x in messages[-1]['content'].split()]
        def decode(self, tokens, **kwargs):
            return 'not JSON'

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(157)
        cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, attention_dropout=0., max_position_embeddings=128,
            bos_token_id=1, eos_token_id=2, pad_token_id=0)
        cfg._attn_implementation = 'eager'
        model = LlamaForCausalLM(cfg).eval().requires_grad_(False)
        before = parameter_digest(model)
        cases = [EvalCase('a', 'math', '1 3', 'tool_call', '1+2', 3),
                 EvalCase('b', 'math', '1 4 5', 'tool_call', '1+2', 3)]
        tok = Tokenizer()
        serial = [predict(model, tok, case) for case in cases]
        batch = predict_case_batch(model, tok, cases)
        assert [r['response_token_ids'] for r in batch] == [r['response_token_ids'] for r in serial]
        assert [r['finish_reason'] for r in batch] == [r['finish_reason'] for r in serial]
        assert parameter_digest(model) == before and all(p.grad is None for p in model.parameters())
    finally:
        torch.set_num_threads(threads)


def test_batch_keeps_real_eos_but_removes_only_post_eos_padding():
    import torch
    from evaluate_reward_training_cases import predict_case_batch
    from agentic_post_training.evaluation.tool_calls import EvalCase

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def apply_chat_template(self, messages, **kwargs):
            return [1, 3] if messages[-1]['content'] == 'short' else [1, 4, 5]
        def decode(self, tokens, **kwargs):
            return 'not JSON'

    class Model:
        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.tolist() == [[0, 1, 3], [1, 4, 5]]
            assert attention_mask.tolist() == [[0, 1, 1], [1, 1, 1]]
            suffix = torch.tensor([[7, 2, 0], [8, 9, 2]])
            return torch.cat([input_ids, suffix], dim=1)

    cases = [EvalCase('a', 'math', 'short', 'tool_call', '1+2', 3),
             EvalCase('b', 'math', 'long', 'tool_call', '1+2', 3)]
    result = predict_case_batch(Model(), Tokenizer(), cases)
    assert [r['response_token_ids'] for r in result] == [[7, 2], [8, 9, 2]]
    assert all(r['finish_reason'] == 'eos' for r in result)
