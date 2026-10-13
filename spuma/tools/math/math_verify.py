"""`math_verify`: check whether two agent-supplied answers are mathematically equivalent (Math-Verify).

The tool only ever sees the two expressions the agent passes in; reference answers are never available
to it (grading lives in `spuma.dataset.math.grading`).
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from math_verify import parse as mv_parse
from math_verify import verify as mv_verify
from pydantic import BaseModel, Field

from spuma.tools.math.expressions import error
from spuma.tools.worker import IsolatedRunner


class VerifyInput(BaseModel):
    expression_a: str = Field(description="First answer or expression (LaTeX allowed)")
    expression_b: str = Field(description="Second answer or expression (LaTeX allowed)")


def verify(expression_a: str, expression_b: str) -> dict[str, Any]:
    """Runs in the isolated worker; never raises."""

    try:
        a, b = mv_parse(_wrap(expression_a)), mv_parse(_wrap(expression_b))
        if not a or not b:
            missing = [n for n, p in (("expression_a", a), ("expression_b", b)) if not p]
            return {"status": "error", "error_type": "parse_error", "error": f"could not parse {missing}"}
        return {"status": "success", "equivalent": bool(mv_verify(a, b)), "method": "symbolic"}
    except Exception as e:  # noqa: BLE001
        return error(e)


def _wrap(text: str) -> str:
    """Math-Verify extracts from LaTeX delimiters; bare answers are wrapped so they parse."""
    text = text.strip()
    return text if ("\\boxed" in text or text.startswith("$")) else f"${text}$"


def build(name: str, runner: IsolatedRunner, timeout: float) -> BaseTool:
    @tool(name, args_schema=VerifyInput)
    def math_verify(expression_a: str, expression_b: str) -> dict[str, Any]:
        """Check whether two mathematical answers or expressions are equivalent (Math-Verify)."""
        return runner.run(verify, timeout, expression_a=expression_a, expression_b=expression_b)

    return math_verify
