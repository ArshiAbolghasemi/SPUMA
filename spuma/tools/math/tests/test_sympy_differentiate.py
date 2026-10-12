from spuma.tools.math.sympy_differentiate import DifferentiateInput, build, differentiate
from spuma.tools.math.tests.conftest import assert_langchain_tool


def test_differentiate():
    assert differentiate("x**3", "x", 2)["result"] == "6*x"
    assert differentiate("x", "1x")["status"] == "error"


def test_tool(runner):
    t = build("sympy_differentiate", runner, timeout=30)
    assert_langchain_tool(t, "sympy_differentiate", DifferentiateInput, "Differentiate")
    assert t.invoke({"expression": "sin(x)", "variable": "x"})["result"] == "cos(x)"
