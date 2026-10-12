import pytest

from spuma.tools.base import ToolUnavailable
from spuma.tools.config import ToolConfig
from spuma.tools.math.provider import SYMPY_TOOLS, VERIFY_TOOLS, MathToolProvider


def test_one_module_per_tool():
    modules = {b.__module__ for b in [*SYMPY_TOOLS.values(), *VERIFY_TOOLS.values()]}
    assert modules == {f"spuma.tools.math.{n}" for n in [*SYMPY_TOOLS, *VERIFY_TOOLS]}


def test_provider_builds_configured_tools(runner):
    provider = MathToolProvider(runner, SYMPY_TOOLS)
    assert provider.build("sympy_solve", ToolConfig(provider="sympy")).name == "sympy_solve"
    # a registry entry may expose an operation under another name
    renamed = provider.build("solve", ToolConfig(provider="sympy", operation="sympy_solve"))
    assert renamed.name == "solve"
    with pytest.raises(ToolUnavailable):
        provider.build("math_verify", ToolConfig(provider="sympy"))  # wrong provider
