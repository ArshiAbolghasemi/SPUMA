import pytest
import sympy

from spuma.tools.math.expressions import ExpressionError, error, ok, parse, symbol


def test_parse_basic():
    assert parse("2x + 3") == 2 * sympy.Symbol("x") + 3
    assert parse("x^2") == sympy.Symbol("x") ** 2
    assert parse("x = 2") == sympy.Symbol("x") - 2
    assert parse("ln(E)") == 1


@pytest.mark.parametrize(
    "bad",
    [
        "__import__('os').system('ls')",
        "x.__class__",
        "lambda: 1",
        "(x for x in y)",
        "open('f')",
        "[1, 2]",
        "a = b = c",
        "",
        "x" * 600,
    ],
)
def test_parse_rejects_code(bad):
    with pytest.raises(ExpressionError):
        parse(bad)


def test_unknown_names_become_symbols_never_builtins():
    assert parse("exec") == sympy.Symbol("exec")
    with pytest.raises(ExpressionError):
        parse("eval(x)")


def test_symbol_and_results():
    assert symbol("x1") == sympy.Symbol("x1")
    with pytest.raises(ExpressionError):
        symbol("1x")
    assert ok(3, latex="3") == {"status": "success", "result": 3, "latex": "3"}
    assert error(ExpressionError("bad"))["error_type"] == "invalid_expression"
    assert error(ZeroDivisionError())["error_type"] == "math_error"
