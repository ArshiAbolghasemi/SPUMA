import dataclasses
import json

import pytest

from spuma.tools.math.sympy_simplify import simplify
from spuma.tools.registry import (
    ToolExecutor,
    build_tool_registry,
    default_providers,
)
from spuma.tools.worker import IsolatedRunner, ToolTimeout


def _run(executor, name, args, allowed, used=0, max_calls=5, call_id="c0"):
    return executor.execute(
        {"name": name, "args": args, "id": call_id},
        problem_id="p1",
        agent_id="solver",
        step_id=0,
        allowed_tools=allowed,
        calls_used=used,
        max_calls=max_calls,
        message_tokens=12,
    )


def test_registration_and_discovery(registry):
    assert registry.enabled_names() == {
        "sympy_solve",
        "sympy_simplify",
        "sympy_integrate",
        "sympy_differentiate",
        "math_verify",
    }
    assert registry.disabled == {}
    (schema,) = registry.schemas(["sympy_solve"])
    assert schema["function"]["name"] == "sympy_solve" and "equations" in schema["function"]["parameters"]["properties"]


def test_effective_tools_is_intersection(registry, config):
    solver, reviewer = config.agents.items
    assert registry.effective_tools(solver.allowed_tools) == sorted(
        ["sympy_solve", "sympy_simplify", "sympy_integrate", "sympy_differentiate"]
    )
    assert "math_verify" in registry.effective_tools(reviewer.allowed_tools)
    assert "math_verify" not in registry.effective_tools(solver.allowed_tools)
    assert registry.effective_tools([]) == []


def test_successful_call_record(registry):
    record = _run(
        ToolExecutor(registry), "sympy_solve", {"equations": ["x**2 = 9"], "variables": ["x"]}, ["sympy_solve"]
    )
    assert record.status == "success" and record.error is None
    assert record.arguments == {"equations": ["x**2 = 9"], "variables": ["x"]}
    assert record.tool_call_message_tokens == 12 and record.duration_s > 0
    assert json.loads(record.observation)["status"] == "success"
    assert set(record.to_dict()) >= {
        "problem_id",
        "agent_id",
        "step_id",
        "tool",
        "arguments",
        "result",
        "status",
        "error",
    }


def test_unauthorized_call_is_rejected_even_when_enabled(registry, mocker):
    spy = mocker.patch.object(registry.tools["math_verify"], "func")
    record = _run(ToolExecutor(registry), "math_verify", {"expression_a": "1", "expression_b": "1"}, ["sympy_solve"])
    assert record.status == "permission_denied" and "not permitted" in record.error
    assert json.loads(record.observation) == {"status": "permission_denied", "error": record.error}
    spy.assert_not_called()


def test_disabled_tool_rejected_even_if_allowed(config, set_settings, runner):
    entries = dict(config.tools.registry)
    entries["sympy_integrate"] = dataclasses.replace(entries["sympy_integrate"], enabled=False)
    set_settings(dataclasses.replace(config, tools=dataclasses.replace(config.tools, registry=entries)))
    reg = build_tool_registry(default_providers(runner))
    assert reg.disabled == {"sympy_integrate": "disabled in config"}
    assert "sympy_integrate" not in reg.effective_tools(["sympy_integrate", "sympy_solve"])
    record = _run(ToolExecutor(reg), "sympy_integrate", {"expression": "x", "variable": "x"}, ["sympy_integrate"])
    assert record.status == "disabled"


def test_runtime_global_disable(config, set_settings, runner):
    set_settings(dataclasses.replace(config, tools=dataclasses.replace(config.tools, **{"enabled": False})))
    reg = build_tool_registry(default_providers(runner))
    assert reg.effective_tools(["sympy_solve"]) == []
    assert _run(ToolExecutor(reg), "sympy_solve", {"equations": ["x"]}, ["sympy_solve"]).status == "disabled"


def test_unknown_tool_budget_and_invalid(registry):
    ex = ToolExecutor(registry)
    assert _run(ex, "rm_rf", {}, ["rm_rf"]).status == "unknown_tool"
    assert _run(ex, "sympy_solve", {"equations": ["x"]}, ["sympy_solve"], used=5).status == "budget_exceeded"
    assert _run(ex, "sympy_solve", {"equations": []}, ["sympy_solve"]).status == "invalid_arguments"
    bad = ex.execute(
        {"name": None, "args": "{oops", "id": "c1", "error": "malformed"},
        problem_id="p",
        agent_id="a",
        step_id=0,
        allowed_tools=[],
        calls_used=0,
        max_calls=5,
    )
    assert bad.status == "invalid_call"


def test_tool_error_is_structured(registry):
    record = _run(ToolExecutor(registry), "sympy_simplify", {"expression": "__import__('os')"}, ["sympy_simplify"])
    assert record.status == "tool_error" and record.result["error_type"] == "invalid_expression"


def _execution(config, set_settings, **update):
    execution = dataclasses.replace(config.tools.execution, **update)
    set_settings(dataclasses.replace(config, tools=dataclasses.replace(config.tools, **{"execution": execution})))


def test_output_truncated_by_tokens(registry, config, set_settings):
    _execution(config, set_settings, max_output_tokens=10)
    ex = ToolExecutor(registry, truncate=lambda t, n: t[:n])
    assert len(_run(ex, "sympy_simplify", {"expression": "x+1"}, ["sympy_simplify"]).observation) == 10


def test_save_results_off(registry, config, set_settings):
    _execution(config, set_settings, save_results=False)
    ex = ToolExecutor(registry)
    record = _run(ex, "sympy_simplify", {"expression": "x+x"}, ["sympy_simplify"])
    assert record.result is None and "2*x" in record.observation


def _slow(seconds: float) -> dict:
    import time

    time.sleep(seconds)
    return {"status": "success"}


def test_worker_timeout_restarts():
    runner = IsolatedRunner()
    try:
        with pytest.raises(ToolTimeout):
            runner.run(_slow, 0.5, seconds=30)
        assert runner.run(simplify, 30, expression="x+x")["result"] == "2*x"
    finally:
        runner.close()


def test_timeout_status_from_executor(registry, mocker):
    mocker.patch.object(registry.tools["sympy_solve"], "func", side_effect=ToolTimeout("slow"))
    assert _run(ToolExecutor(registry), "sympy_solve", {"equations": ["x"]}, ["sympy_solve"]).status == "timeout"
