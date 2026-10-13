import pytest

from spuma.tools.base import ToolUnavailable
from spuma.tools.config import ToolConfig
from spuma.tools.math.provider import SYMPY_TOOLS, VERIFY_TOOLS, MathToolProvider


def test_one_module_per_tool():
    modules = {b.__module__ for b in [*SYMPY_TOOLS.values(), *VERIFY_TOOLS.values()]}
    assert modules == {f"spuma.tools.math.{n}" for n in [*SYMPY_TOOLS, *VERIFY_TOOLS]}


def test_provider_builds_configured_tools(runner, config, set_settings):
    # a registry entry may expose an operation under another name
    entries = {**config.tools.registry, "solve": ToolConfig(provider="sympy", operation="sympy_solve")}
    set_settings(config.replace(tools=config.tools.model_copy(update={"registry": entries})))
    provider = MathToolProvider(runner, SYMPY_TOOLS)
    assert provider.build("sympy_solve").name == "sympy_solve"
    assert provider.build("solve").name == "solve"
    with pytest.raises(ToolUnavailable):
        provider.build("math_verify")  # registered, but not a SymPy operation
