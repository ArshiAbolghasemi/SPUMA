from spuma.tools.math.sympy_solve import SolveInput, build, solve
from spuma.tools.math.tests.conftest import assert_langchain_tool


def test_solve():
    out = solve(["x**2 = 4"], ["x"])
    assert out["status"] == "success" and {s["x"] for s in out["result"]} == {"-2", "2"}
    assert solve(["x + y = 3", "x - y = 1"])["result"] == [{"x": "2", "y": "1"}]
    assert solve(["import os"])["error_type"] == "invalid_expression"


def test_tool(runner):
    t = build("sympy_solve", runner, timeout=30)
    assert_langchain_tool(t, "sympy_solve", SolveInput, "Solve an equation")
    assert t.invoke({"equations": ["2x + 3 = 7"], "variables": ["x"]})["result"] == [{"x": "2"}]
