from spuma.tools.math.sympy_integrate import IntegrateInput, build, integrate
from spuma.tools.math.tests.conftest import assert_langchain_tool


def test_integrate():
    assert integrate("2x", "x")["result"] == "x**2"
    out = integrate("x", "x", "0", "2")
    assert out["result"] == "2" and out["definite"] is True
    assert integrate("x", "x", "0")["status"] == "error"  # one bound only


def test_tool(runner):
    t = build("sympy_integrate", runner, timeout=30)
    assert_langchain_tool(t, "sympy_integrate", IntegrateInput, "Integrate")
    assert t.invoke({"expression": "3x^2", "variable": "x", "lower": "0", "upper": "1"})["result"] == "1"
