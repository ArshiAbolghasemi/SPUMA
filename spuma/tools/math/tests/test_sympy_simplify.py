from spuma.tools.math.sympy_simplify import SimplifyInput, build, simplify
from spuma.tools.math.tests.conftest import assert_langchain_tool


def test_simplify():
    assert simplify("(x**2 - 1)/(x - 1)")["result"] == "x + 1"
    out = simplify("__import__('os')")
    assert out["status"] == "error" and out["error_type"] == "invalid_expression"


def test_tool(runner):
    t = build("sympy_simplify", runner, timeout=30)
    assert_langchain_tool(t, "sympy_simplify", SimplifyInput, "Simplify")
    assert t.invoke({"expression": "x + x"})["result"] == "2*x"
