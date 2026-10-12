"""Mathematical tools, one module per LangChain tool (`langchain_core.tools.tool`):

    sympy_solve, sympy_simplify, sympy_integrate, sympy_differentiate   (SymPy)
    math_verify                                                         (Math-Verify)

Each module holds its input schema, the operation (run in the isolated worker) and `build(...)`, which
returns the decorated tool. `expressions` is the shared safe parser; `provider` wires them to the registry.
"""
