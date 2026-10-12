from spuma.tools.math.math_verify import VerifyInput, build, verify
from spuma.tools.math.tests.conftest import assert_langchain_tool


def test_verify():
    assert verify("\\frac{1}{2}", "0.5") == {"status": "success", "equivalent": True, "method": "symbolic"}
    assert verify("2", "3")["equivalent"] is False
    assert verify("\\boxed{x+1}", "1+x")["equivalent"] is True


def test_tool(runner):
    t = build("math_verify", runner, timeout=30)
    assert_langchain_tool(t, "math_verify", VerifyInput, "Check whether two")
    assert t.invoke({"expression_a": "\\sqrt{4}", "expression_b": "2"})["equivalent"] is True
