"""Offline arithmetic grading v2; deliberately separate from the frozen trainer contract."""

import ast
from collections import Counter
import json

from agentic_post_training.environments.calculator import parse_expression
from agentic_post_training.rewards.contract import score_reward

CONTRACT = 'arithmetic-graded-v2-offline'
LEVELS = {'invalid': 0., 'executable': .1, 'components_match': .3, 'verified_correct': 1.}


def components(tree):
    """Count numeric leaves and all binary/unary operators, including repetitions."""
    return (Counter(float(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)),
            Counter(type(n.op).__name__ for n in ast.walk(tree) if isinstance(n, (ast.BinOp, ast.UnaryOp))))


def commutative_structure(node):
    """Allow child swaps at Add/Mult only; preserve grouping, subtraction and division."""
    if isinstance(node, ast.Expression):
        return commutative_structure(node.body)
    if isinstance(node, ast.Constant):
        return ('number', float(node.value))
    if isinstance(node, ast.UnaryOp):
        return (type(node.op).__name__, commutative_structure(node.operand))
    if isinstance(node, ast.BinOp):
        children = [commutative_structure(node.left), commutative_structure(node.right)]
        if isinstance(node.op, (ast.Add, ast.Mult)):
            children.sort(key=repr)
        return (type(node.op).__name__, *children)
    raise ValueError('unsupported arithmetic AST')


def grade_answer(text, case, *, finish_reason):
    """Award one mutually exclusive tier, never a sum of independent format bonuses."""
    old = score_reward(text, case, finish_reason=finish_reason)
    executable = bool(old['eligible'] and finish_reason == 'eos' and old['score']['schema_valid']
                      and old['score']['route_correct'] and old['score']['tool_execution_success'])
    component_match = equivalent = False
    if executable:
        actual = parse_expression(json.loads(text)['arguments']['expression'])
        expected = parse_expression(case.expected_expression)
        component_match = components(actual) == components(expected)
        equivalent = component_match and commutative_structure(actual) == commutative_structure(expected)
    full = bool(executable and old['score']['result_correct'] and (old['reward'] == 1. or equivalent))
    tier = ('verified_correct' if full else 'components_match' if component_match
            else 'executable' if executable else 'invalid')
    return {'graded_contract_version': CONTRACT, 'graded_reward': LEVELS[tier], 'graded_tier': tier,
            'graded_eligible': old['eligible'], 'strict_reward': old['reward'],
            'verified_equivalent_reward': float(full),
            'graded_checks': {'completed_executable_call': executable, 'number_operator_multisets_match': component_match,
                              'commutative_ast_matches': equivalent, 'numeric_result_correct': old['score']['result_correct']},
            'training_admitted': False}
