"""`sympy_differentiate`: differentiate an expression with SymPy."""

from typing import Any

import sympy
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from spuma.tools.math.expressions import error, ok, parse, symbol
from spuma.tools.worker import IsolatedRunner


class DifferentiateInput(BaseModel):
    expression: str = Field(description="Expression to differentiate")
    variable: str = Field(description="Differentiation variable")
    order: int = Field(1, ge=1, le=10, description="Derivative order")


def differentiate(expression: str, variable: str, order: int = 1) -> dict[str, Any]:
    """Runs in the isolated worker; never raises."""
    try:
        out = sympy.diff(parse(expression), symbol(variable), order)
        return ok(str(out), latex=sympy.latex(out))
    except Exception as e:  # noqa: BLE001
        return error(e)


def build(name: str, runner: IsolatedRunner, timeout: float) -> BaseTool:
    @tool(name, args_schema=DifferentiateInput)
    def sympy_differentiate(expression: str, variable: str, order: int = 1) -> dict[str, Any]:
        """Differentiate an expression with SymPy."""
        return runner.run(differentiate, timeout, expression=expression, variable=variable, order=order)

    return sympy_differentiate
