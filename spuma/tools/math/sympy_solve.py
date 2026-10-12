"""`sympy_solve`: solve an equation or a system of equations symbolically with SymPy."""

from typing import Any

import sympy
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from spuma.tools.math.expressions import error, ok, parse, symbol
from spuma.tools.worker import IsolatedRunner


class SolveInput(BaseModel):
    equations: list[str] = Field(
        min_length=1, max_length=10, description="Equations like 'x**2 = 4' or expressions = 0"
    )
    variables: list[str] | None = Field(None, max_length=10, description="Variables to solve for")


def solve(equations: list[str], variables: list[str] | None = None) -> dict[str, Any]:
    """Runs in the isolated worker; never raises."""
    try:
        exprs = [parse(e) for e in equations]
        syms = [symbol(v) for v in variables] if variables else None
        sols = sympy.solve(exprs, syms, dict=True) if syms else sympy.solve(exprs, dict=True)
        return ok([{str(k): str(v) for k, v in s.items()} for s in sols])
    except Exception as e:  # noqa: BLE001
        return error(e)


def build(name: str, runner: IsolatedRunner, timeout: float) -> BaseTool:
    @tool(name, args_schema=SolveInput)
    def sympy_solve(equations: list[str], variables: list[str] | None = None) -> dict[str, Any]:
        """Solve an equation or a system of equations symbolically with SymPy."""
        return runner.run(solve, timeout, equations=equations, variables=variables)

    return sympy_solve
