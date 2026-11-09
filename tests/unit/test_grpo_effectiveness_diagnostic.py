import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from diagnose_grpo_effectiveness import first_difference, score_completion, probability_changes, error_kind
from agentic_post_training.evaluation.tool_calls import EvalCase
from agentic_post_training.rewards.contract import score_reward


def test_first_difference_requires_complete_shared_prefix_comparison():
    assert first_difference([4, 5, 2], [4, 6, 2]) == 1
    assert first_difference([4, 2], [4, 2]) is None
    with pytest.raises(ValueError, match='strict prefix'):
        first_difference([4], [4, 2])


def test_teacher_forced_probe_shifts_once_and_compares_same_prefix_without_gradients():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(6 * 8, dtype=torch.float32).reshape(6, 8) / 10)
        def forward(self, input_ids, attention_mask, use_cache):
            assert not torch.is_grad_enabled() and not use_cache
            return SimpleNamespace(logits=self.table[:input_ids.shape[1]].unsqueeze(0))
    model = Model().eval()
    result = score_completion(model, [1, 3], [4, 5, 2], [4, 6, 2])
    expected = model.table.detach().log_softmax(-1)
    values = torch.tensor([expected[1, 4], expected[2, 5], expected[3, 2]])
    assert result['sequence_logp'] == pytest.approx(values.double().sum().item())
    assert result['mean_token_logp'] == pytest.approx(values.double().mean().item())
    event = result['first_difference']
    assert event['response_index'] == 1 and event['correct_token_id'] == 5
    assert event['gap_correct_minus_old_wrong'] == pytest.approx((expected[2, 5] - expected[2, 6]).item())
    assert event['correct_token_rank'] == 3 and model.table.grad is None


@pytest.mark.parametrize('expression,expected', [
    ('3*(1+2)', 'passed'),
    ('(1+2)*3', 'value_correct_ast_mismatch'),
    ('3*(1+1)', 'wrong_number_multiset'),
    ('3+(1+2)', 'wrong_operator_multiset'),
    ('3*1+2', 'wrong_order_or_grouping'),
])
def test_error_types_separate_semantic_equivalence_from_wrong_arithmetic(expression, expected):
    import json
    case = EvalCase('a', 'math', 'x', 'tool_call', '3*(1+2)', 9)
    text = json.dumps({'type': 'tool_call', 'name': 'calculator', 'arguments': {'expression': expression}})
    row = {'text': text, 'finish_reason': 'eos', **score_reward(text, case, finish_reason='eos')}
    assert error_kind(row, case) == expected


def test_probability_summary_does_not_confuse_absolute_and_relative_improvement():
    before = {'canonical': {'sequence_logp': -3., 'first_difference': {
        'gap_correct_minus_old_wrong': -.5, 'correct_token_rank': 2}}, 'wrong': {'sequence_logp': -2.}}
    after = copy.deepcopy(before)
    after['canonical']['sequence_logp'] = -2.8  # Correct answer probability rises.
    after['wrong']['sequence_logp'] = -1.5      # Wrong answer rises more.
    after['canonical']['first_difference']['gap_correct_minus_old_wrong'] = -.6
    result = probability_changes([{'suite': 'development', 'wording': 'natural',
                                   'models': {'a': before, 'b': after}}], 'a', 'b')['development / natural']
    assert result['canonical_logp_up'] == 1
    assert result['sequence_margin_up'] == 0 and result['first_difference_gap_up'] == 0
    assert result['mean_correct_wrong_sequence_margin_delta'] == pytest.approx(-.3)
