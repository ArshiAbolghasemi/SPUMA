"""Shared by the SymPy tools: safe parsing of untrusted expressions and structured results.

Untrusted expressions never reach `eval` unchecked: `parse` accepts only a whitelisted character
set (no quotes, brackets, underscores or attribute access), rejects Python keywords, and evaluates
with a global namespace holding nothing but whitelisted SymPy objects (unknown names become
Symbols). Operations call it inside the isolated worker process (`spuma/tools/worker.py`).
"""

import keyword
import re
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

MAX_EXPRESSION_CHARS = 500
_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9\s+\-*/^().,=!]*$")
_BAD_DOT = re.compile(r"(?<![0-9])\.|\.(?![0-9])")  # dots only inside numbers: no attribute access
_IDENT = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_TRANSFORMS = (*standard_transformations, implicit_multiplication, convert_xor)
_NAMES: dict[str, Any] = {
    n: getattr(sympy, n)
    for n in (
        "sin cos tan cot sec csc asin acos atan atan2 sinh cosh tanh asinh acosh atanh exp log sqrt cbrt root "
        "Abs floor ceiling factorial binomial gcd lcm pi E I oo Rational Integer Float Symbol re im conjugate "
        "arg Min Max Mod sign"
    ).split()
}
_NAMES |= {"ln": sympy.log, "abs": sympy.Abs}


class ExpressionError(ValueError):
    pass


def parse(text: str) -> sympy.Expr:
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("expression must be a non-empty string")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    if not _ALLOWED_CHARS.match(text):
        raise ExpressionError("expression contains disallowed characters")
    if _BAD_DOT.search(text):
        raise ExpressionError("'.' is only allowed inside numbers")
    if bad := [w for w in _IDENT.findall(text) if keyword.iskeyword(w)]:
        raise ExpressionError(f"disallowed identifiers: {bad}")
    if "=" in text:
        parts = text.split("=")
        if len(parts) != 2:
            raise ExpressionError("at most one '=' is allowed")
        return parse(parts[0]) - parse(parts[1])  # `a = b` means a - b = 0
    try:
        expr = parse_expr(text, local_dict={}, global_dict={"__builtins__": {}, **_NAMES}, transformations=_TRANSFORMS)
    except Exception as e:  # noqa: BLE001 - any parser failure is an invalid expression
        raise ExpressionError(f"could not parse {text!r}: {type(e).__name__}") from None
    if not isinstance(expr, sympy.Basic):
        raise ExpressionError(f"{text!r} is not a mathematical expression")
    return expr


def symbol(name: str) -> sympy.Symbol:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name or ""):
        raise ExpressionError(f"invalid variable name {name!r}")
    return sympy.Symbol(name)


def ok(result: Any, **extra: Any) -> dict[str, Any]:
    return {"status": "success", "result": result, **extra}


def error(e: Exception) -> dict[str, Any]:
    kind = "invalid_expression" if isinstance(e, ExpressionError) else "math_error"
    return {"status": "error", "error_type": kind, "error": str(e) or type(e).__name__}
