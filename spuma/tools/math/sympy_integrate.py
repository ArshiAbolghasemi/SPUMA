"""`sympy_integrate`: integrate an expression with SymPy (definite when both bounds are given)."""

from typing import Any

import sympy
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from spuma.tools.math.expressions import ExpressionError, error, ok, parse, symbol
from spuma.tools.worker import IsolatedRunner


class IntegrateInput(BaseModel):
    expression: str = Field(description="Integrand")
    variable: str = Field(description="Integration variable")
    lower: str | None = Field(None, description="Lower bound (definite integral)")
    upper: str | None = Field(None, description="Upper bound (definite integral)")


def integrate(expression: str, variable: str, lower: str | None = None, upper: str | None = None) -> dict[str, Any]:
    """Runs in the isolated worker; never raises."""
    try:
        expr, x = parse(expression), symbol(variable)
        if (lower is None) != (upper is None):
            raise ExpressionError("give both lower and upper bounds, or neither")
        out = sympy.integrate(expr, (x, parse(lower), parse(upper))) if lower is not None else sympy.integrate(expr, x)
        return ok(str(out), latex=sympy.latex(out), definite=lower is not None)
    except Exception as e:  # noqa: BLE001
        return error(e)


def build(name: str, runner: IsolatedRunner, timeout: float) -> BaseTool:
    @tool(name, args_schema=IntegrateInput)
    def sympy_integrate(
        expression: str, variable: str, lower: str | None = None, upper: str | None = None
    ) -> dict[str, Any]:
        """Integrate an expression with SymPy (definite when both bounds are given)."""
        return runner.run(integrate, timeout, expression=expression, variable=variable, lower=lower, upper=upper)

    return sympy_integrate
