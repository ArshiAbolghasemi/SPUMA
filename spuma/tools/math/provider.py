"""`provider: sympy` / `provider: math_verify` registry entries -> the LangChain tools in this package."""

from collections.abc import Callable

from langchain_core.tools import BaseTool

from spuma.tools.base import ToolUnavailable
from spuma.tools.config import ToolConfig
from spuma.tools.math import math_verify, sympy_differentiate, sympy_integrate, sympy_simplify, sympy_solve
from spuma.tools.worker import IsolatedRunner

ToolBuilder = Callable[[str, IsolatedRunner, float], BaseTool]

# operation name (the registry key, or its `operation:` option) -> the module's `build`
SYMPY_TOOLS: dict[str, ToolBuilder] = {
    "sympy_solve": sympy_solve.build,
    "sympy_simplify": sympy_simplify.build,
    "sympy_integrate": sympy_integrate.build,
    "sympy_differentiate": sympy_differentiate.build,
}
VERIFY_TOOLS: dict[str, ToolBuilder] = {"math_verify": math_verify.build}


class MathToolProvider:
    """Builds the configured math tools; every call runs in the shared isolated worker with its timeout."""

    def __init__(self, runner: IsolatedRunner, tools: dict[str, ToolBuilder]) -> None:
        self.runner, self.tools = runner, tools

    def build(self, name: str, config: ToolConfig) -> BaseTool:
        operation = config.options.get("operation", name)
        if operation not in self.tools:
            raise ToolUnavailable(f"provider {config.provider!r} has no operation {operation!r}")
        return self.tools[operation](name, self.runner, config.timeout_seconds)
