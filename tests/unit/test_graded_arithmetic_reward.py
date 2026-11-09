import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from graded_arithmetic_reward import grade_answer
from rescore_grpo_rewards import group_statistics, aggregate
from agentic_post_training.evaluation.tool_calls import EvalCase


def answer(expression):
    return json.dumps({'type': 'tool_call', 'name': 'calculator', 'arguments': {'expression': expression}})


CASE = EvalCase('a', 'math', 'calculate', 'tool_call', '12*(47+20)', 804)


@pytest.mark.parametrize('expression,reward,tier', [
    ('12*(47+20)', 1., 'verified_correct'),
    ('(47+20)*12', 1., 'verified_correct'),
    ('(20+47)*12', 1., 'verified_correct'),
    ('12.0*(47+20)', 1., 'verified_correct'),
    ('12*47+20', .3, 'components_match'),
    ('12*(47-20)', .1, 'executable'),
    ('47+20', .1, 'executable'),
    ('804', .1, 'executable'),  # Correct number alone is not evidence of the requested calculation.
    ('12*(47+20)+0', .1, 'executable'),
    ('12*(47+20)/0', 0., 'invalid'),
    ('__import__("os")', 0., 'invalid'),
])
def test_gated_tiers_distinguish_correctness_components_and_executability(expression, reward, tier):
    result = grade_answer(answer(expression), CASE, finish_reason='eos')
    assert result['graded_reward'] == reward and result['graded_tier'] == tier
    assert not result['training_admitted']


def test_commutativity_does_not_reassociate_or_swap_subtraction_or_division():
    division = EvalCase('b', 'math', 'calculate', 'tool_call', '8/(4/2)', 4)
    assert grade_answer(answer('(8/4)/2'), division, finish_reason='eos')['graded_reward'] == .3
    subtract = EvalCase('c', 'math', 'calculate', 'tool_call', '8-(4-2)', 6)
    assert grade_answer(answer('(8-4)-2'), subtract, finish_reason='eos')['graded_reward'] == .3
    # Equal final value plus identical leaves/operators does not prove equivalence under the allowed rewrites.
    addition = EvalCase('d', 'math', 'calculate', 'tool_call', '1+(2+3)', 6)
    result = grade_answer(answer('(1+2)+3'), addition, finish_reason='eos')
    assert result['graded_checks']['numeric_result_correct'] and result['graded_reward'] == .3


@pytest.mark.parametrize('text', [
    'not json',
    '{"type":"tool_call","type":"tool_call","name":"calculator","arguments":{"expression":"12*(47+20)"}}',
    '{"type":"tool_call","name":"other","arguments":{"expression":"12*(47+20)"}}',
    '{"type":"final","answer":"804"}',
    '{"type":"tool_call","name":"calculator","arguments":{"expression":"12*(47+20)","extra":0}}',
])
def test_invalid_format_route_and_schema_cannot_earn_partial_points(text):
    assert grade_answer(text, CASE, finish_reason='eos')['graded_reward'] == 0.


def test_truncation_never_earns_points_even_for_complete_correct_json():
    result = grade_answer(answer('12*(47+20)'), CASE, finish_reason='length')
    assert result['graded_reward'] == result['strict_reward'] == 0.
    greeting = EvalCase('g', 'greet', 'hello', 'final')
    result = grade_answer('{"type":"final","answer":"Hello"}', greeting, finish_reason='eos')
    assert not result['graded_eligible'] and result['graded_reward'] == 0.


def test_group_signal_uses_actual_normalized_advantages_and_zero_variance():
    rows = [{'sample_index': i, 'strict_reward': 0., 'verified_equivalent_reward': 0., 'graded_reward': .1}
            for i in range(8)]
    assert group_statistics(rows)['graded']['advantages'] == [0.] * 8
    rows[0]['graded_reward'] = .3
    statistics = group_statistics(rows)
    assert statistics['graded']['has_relative_signal']
    assert statistics['graded']['advantages'][0] > 0 and statistics['graded']['advantages'][1] < 0
    assert statistics['graded']['nonzero_advantage_samples'] == 8
    report = aggregate([{'samples': 8, 'statistics': statistics}])
    assert report['partial_only_signal_groups'] == 1 and report['groups_with_verified_full_credit'] == 0
    assert report['new_signal_equivalent_to_graded'] == 1
    with pytest.raises(ValueError, match='complete ordered'):
        group_statistics(rows[:-1] + [rows[0]])


def test_new_acceptance_can_remove_variance_when_all_answers_become_correct():
    rows = [{'sample_index': 0, 'strict_reward': 1., 'verified_equivalent_reward': 1., 'graded_reward': 1.},
            {'sample_index': 1, 'strict_reward': 0., 'verified_equivalent_reward': 1., 'graded_reward': 1.}]
    report = aggregate([{'samples': 2, 'statistics': group_statistics(rows)}])
    assert report['lost_signal_strict_to_equivalent'] == 1
    assert report['still_zero_variance_groups'] == 1 and report['groups_with_verified_full_credit'] == 1
