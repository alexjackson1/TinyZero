import ast
import operator
import re
import random
from typing import Any, List, Optional, TypedDict


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
NUMBER_PATTERN = re.compile(r"\d+")
ALLOWED_PATTERN = r"^[\d+\-*/×÷\(\)\s]+$"

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def extract_expression(response_text: str) -> str:
    """Extract the equation from the solution string."""

    # Remove everything before the first "Assistant:"
    if "Assistant:" in response_text:
        response = response_text.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in response_text:
        response = response_text.split("<|im_start|>assistant", 1)[1]
    else:
        return None

    # Extract the last line of the response
    last_line = response.split("\n")[-1]

    # Extract the answer from the last line
    matches = ANSWER_PATTERN.findall(last_line)
    return matches[-1].strip() if matches else None


def check_constant(value: Any) -> int:
    """
    Ensure that the value is a non-negative integer.
    If the value is a float that is an integer (e.g., 3.0), it is converted to int.
    Otherwise, raise ValueError.
    """
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            if value.is_integer():
                value = int(value)
            else:
                raise ValueError("Non-integer value encountered")
        if value < 0:
            raise ValueError("Negative value encountered")
        return value
    else:
        raise ValueError("Unsupported value type")


def validate_expression(expression: str, available_numbers: List[int]):
    """Validate that equation only uses available numbers and each number at most once."""
    try:
        # Extract all natural numbers from the equation
        numbers = [int(n) for n in NUMBER_PATTERN.findall(expression)]

        # Check that each number is used at most once
        unused = list(available_numbers)
        for n in numbers:
            if n not in unused:
                return False
            unused.remove(n)

        return True
    except:
        return False


def evaluate_expression(expression: str) -> Optional[int]:
    """
    Evaluate an arithmetic expression safely, ensuring that all intermediate
    results are non-negative integers.
    """
    if not re.match(ALLOWED_PATTERN, expression):
        return None

    # Replace alternate operator symbols with Python's operators.
    expression = expression.replace("×", "*").replace("÷", "/")

    # Parse the expression into an AST.
    try:
        parsed_expr = ast.parse(expression, mode="eval")
    except SyntaxError as err:
        return None

    try:
        result = _eval_ast(parsed_expr.body)
        return result
    except ValueError:
        return None


def _eval_ast(node: ast.AST) -> int:
    """Recursively evaluate an AST node ensuring all results are non-negative integers."""
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        op_type = type(node.op)
        if op_type in OPERATORS:
            result = OPERATORS[op_type](left, right)
            return check_constant(result)
        else:
            raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_ast(node.operand)
        op_type = type(node.op)
        if op_type in OPERATORS:
            result = OPERATORS[op_type](operand)
            return check_constant(result)
        else:
            raise ValueError(f"Unsupported unary operator: {op_type}")
    elif isinstance(node, ast.Constant):  # NOTE: this will break if not Py 3.8+
        if isinstance(node.value, (int, float)):
            return check_constant(node.value)
        else:
            raise ValueError("Unsupported constant type")
    else:
        raise ValueError(f"Unsupported AST node: {node}")


class GroundTruth(TypedDict):
    target: int
    starting: List[int]
    closest: int


def compute_score(
    response_text: str,
    ground_truth: GroundTruth,
    format_score: float = 0.1,
    score: float = 1.0,
):
    """The scoring function for countdown task.

    Args:
        response_text: the response text
        ground_truth: dictionary containing target number, starting numbers, and
            closest value to target that can be made with the starting numbers
        method: unused
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    starting = ground_truth["numbers"]
    closest = ground_truth["closest"]

    expression = extract_expression(response_text)
    should_print = random.randint(1, 64) == 1

    if should_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {starting}")
        print(f"Extracted expression: {expression}")
        print(f"Solution string: {response_text}")

    if expression is None:
        if should_print:
            print(f"No expression found")
        return 0

    # Check formatting
    if not validate_expression(expression, starting):
        if should_print:
            print(f"Invalid expression format")
        return format_score

    # Check expression result
    try:
        result = evaluate_expression(expression)
        if result is None:
            if should_print:
                print(f"Could not evaluate equation")
            return format_score

        if (result - closest) == 0:  # Account for floating point precision
            if should_print:
                print(f"Correct equation: {expression} = {result} (target = {target})")
            return score
        else:
            if should_print:
                print(
                    f"Wrong result: equation = {result}, target = {target} (closest = {closest})"
                )
            return format_score
    except:
        if should_print:
            print(f"Error evaluating equation")
        return format_score


if __name__ == "__main__":
    examples = [
        "2 + 3 * (4 - 1)",  # Valid, result = 11
        "3 - 5",  # Invalid: intermediate result negative (3-5 = -2)
        "6 ÷ 4",  # Invalid: division yields non-integer (1.5)
        "10 - 2 * 5",  # Valid: 10 - (2*5) = 0
        "(48/2+2-4)/2",  # Valid, result = 11
    ]

    for expr in examples:
        try:
            result = evaluate_expression(expr)
            print(f"The result of '{expr}' is {result}")
        except ValueError as e:
            print(f"Expression '{expr}' failed: {e}")
