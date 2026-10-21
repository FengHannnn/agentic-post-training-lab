"""Bounded arithmetic interpreted from AST, never Python eval."""

import ast
import math
import operator


OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def parse_expression(expression: str) -> ast.Expression:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 128:
        raise ValueError("expression must contain 1..128 characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("expression is too complex")
    return tree


def calculate(expression: str) -> float:
    tree = parse_expression(expression)

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = float(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand)
            result = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            result = OPERATORS[type(node.op)](visit(node.left), visit(node.right))
        else:
            raise ValueError("only numbers, +, -, *, / and parentheses are allowed")
        if not math.isfinite(result) or abs(result) > 1e12:
            raise ValueError("arithmetic result exceeds finite bounds")
        return result

    try:
        return visit(tree.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError("division by zero or numeric overflow") from error


def expression_structure(expression: str) -> str:
    calculate(expression)
    return ast.dump(parse_expression(expression), include_attributes=False)
