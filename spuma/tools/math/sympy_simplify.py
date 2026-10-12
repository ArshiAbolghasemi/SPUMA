"""`sympy_simplify`: simplify a mathematical expression with SymPy."""

from typing import Any

import sympy
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from spuma.tools.math.expressions import error, ok, parse
from spuma.tools.worker import IsolatedRunner


class SimplifyInput(BaseModel):
    expression: str = Field(description="Expression to simplify, e.g. '(x**2 - 1)/(x - 1)'")


def simplify(expression: str) -> dict[str, Any]:
    """Runs in the isolated worker; never raises."""
    try:
        out = sympy.simplify(parse(expression))
        return ok(str(out), latex=sympy.latex(out))
    except Exception as e:  # noqa: BLE001
        return error(e)


def build(name: str, runner: IsolatedRunner, timeout: float) -> BaseTool:
    @tool(name, args_schema=SimplifyInput)
    def sympy_simplify(expression: str) -> dict[str, Any]:
        """Simplify a mathematical expression with SymPy."""
        return runner.run(simplify, timeout, expression=expression)

    return sympy_simplify
