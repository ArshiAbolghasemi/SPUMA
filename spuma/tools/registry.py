"""Tool registry, provider wiring, and the permission-enforcing executor.

Tools are LangChain tools (`langchain_core.tools.tool`) built by providers: `spuma.tools.math` (SymPy,
Math-Verify). The registry exposes their OpenAI-style schemas
to agents; the executor validates and invokes them.

    effective_tools(agent) = enabled registered tools ∩ agent.allowed_tools

Agents are only *shown* their effective tools, but that is not the security boundary: the
`ToolExecutor` re-checks registration, enablement, the agent's allowlist and its call budget on
every invocation, so a hand-written tool-call payload for any other tool gets a structured error.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from spuma.config import get_settings
from spuma.tools.base import ToolProvider, ToolUnavailable
from spuma.tools.math.provider import SYMPY_TOOLS, VERIFY_TOOLS, MathToolProvider
from spuma.tools.worker import IsolatedRunner, ToolTimeout

logger = logging.getLogger(__name__)


def default_providers(runner: IsolatedRunner) -> dict[str, ToolProvider]:
    """`provider:` names usable in tools.registry -> the provider that builds their LangChain tools."""
    return {
        "sympy": MathToolProvider(runner, SYMPY_TOOLS),
        "math_verify": MathToolProvider(runner, VERIFY_TOOLS),
    }


# --- registry ------------------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    tools: dict[str, BaseTool] = field(default_factory=dict)
    disabled: dict[str, str] = field(default_factory=dict)  # name -> reason (config or missing capability)
    globally_enabled: bool = True

    def is_registered(self, name: str) -> bool:
        return name in self.tools or name in self.disabled

    def is_enabled(self, name: str) -> bool:
        return self.globally_enabled and name in self.tools

    def enabled_names(self) -> set[str]:
        return set(self.tools) if self.globally_enabled else set()

    def effective_tools(self, allowed: list[str] | set[str] | frozenset[str]) -> list[str]:
        return sorted(self.enabled_names() & set(allowed))

    def schemas(self, names: list[str]) -> list[dict[str, Any]]:
        return [convert_to_openai_tool(self.tools[n]) for n in names]


def build_tool_registry(providers: dict[str, ToolProvider]) -> ToolRegistry:
    """Every `tools.registry` entry built by its provider (or recorded as disabled, with the reason)."""
    config = get_settings().tools
    registry = ToolRegistry(globally_enabled=config.enabled)
    for name, tool_cfg in config.registry.items():
        if not tool_cfg.enabled:
            registry.disabled[name] = "disabled in config"
            continue
        provider = providers.get(tool_cfg.provider)
        if provider is None:
            raise ValueError(f"tools.registry.{name}: unknown provider {tool_cfg.provider!r}")
        try:
            registry.tools[name] = provider.build(name)
        except ToolUnavailable as e:
            registry.disabled[name] = str(e)
            logger.warning("tool disabled: missing capability", extra={"tool": name, "reason": str(e)})
    logger.info(
        "tool registry built",
        extra={"enabled": sorted(registry.enabled_names()), "disabled": registry.disabled, "global": config.enabled},
    )
    return registry


# --- execution -----------------------------------------------------------------------------------


@dataclass
class ToolRecord:
    problem_id: str
    agent_id: str
    step_id: int
    call_id: str
    tool: str | None
    arguments: Any  # validated args on success; the raw payload otherwise
    # success | tool_error | invalid_call | unknown_tool | disabled | permission_denied | budget_exceeded
    # | invalid_arguments | timeout | error
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None
    duration_s: float = 0.0
    tool_call_message_tokens: int | None = None  # generated tokens of the assistant message that issued the call
    observation: str = ""  # exactly what the model receives back (a ToolMessage), never model-generated

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        truncate: Callable[[str, int], str] | None = None,
    ) -> None:
        self.registry = registry
        self.config = get_settings().tools.execution
        self.truncate = truncate or (lambda text, n: text)

    def execute(
        self,
        call: dict[str, Any],
        *,
        problem_id: str,
        agent_id: str,
        step_id: int,
        allowed_tools: list[str],
        calls_used: int,
        max_calls: int,
        message_tokens: int | None = None,
    ) -> ToolRecord:
        name = call.get("name")
        record = ToolRecord(
            problem_id=problem_id,
            agent_id=agent_id,
            step_id=step_id,
            call_id=call.get("id", ""),
            tool=name,
            arguments=call.get("args"),
            status="pending",
            tool_call_message_tokens=message_tokens,
        )
        start = time.perf_counter()
        if call.get("error"):
            self._fail(record, "invalid_call", call["error"])
        elif not name or not self.registry.is_registered(name):
            self._fail(record, "unknown_tool", f"tool {name!r} is not registered")
        elif not self.registry.is_enabled(name):
            reason = (
                "tools are globally disabled" if not self.registry.globally_enabled else self.registry.disabled[name]
            )
            self._fail(record, "disabled", f"tool {name!r} is disabled: {reason}")
        elif name not in allowed_tools:
            self._fail(record, "permission_denied", f"agent {agent_id!r} is not permitted to call {name!r}")
        elif calls_used >= max_calls:
            self._fail(record, "budget_exceeded", f"agent {agent_id!r} exhausted its {max_calls} tool calls")
        else:
            self._invoke(record, self.registry.tools[name], call.get("args") or {})
        record.duration_s = time.perf_counter() - start
        payload = record.result if record.status == "success" else {"status": record.status, "error": record.error}
        record.observation = self.truncate(json.dumps(payload, default=str), self.config.max_output_tokens)
        if self.config.log_calls:
            logger.info(
                "tool call",
                extra={
                    "problem_id": problem_id,
                    "agent_id": agent_id,
                    "step_id": step_id,
                    "tool": name,
                    "status": record.status,
                    "duration_s": round(record.duration_s, 4),
                },
            )
        if not self.config.save_results:
            record.result = None
        return record

    def _invoke(self, record: ToolRecord, tool: BaseTool, args: dict[str, Any]) -> None:
        try:
            validated = tool.args_schema.model_validate(args)  # type: ignore[union-attr]
        except ValidationError as e:
            self._fail(record, "invalid_arguments", e.errors(include_url=False).__repr__())
            return
        record.arguments = validated.model_dump()
        try:
            result = tool.invoke(record.arguments)
        except ToolTimeout as e:
            self._fail(record, "timeout", str(e))
            return
        except Exception as e:  # noqa: BLE001 - tool failures are reported to the model, not fatal
            if self.config.on_error == "raise":
                raise
            self._fail(record, "error", f"{type(e).__name__}: {e}")
            return
        record.result = result
        if result.get("status") == "success":
            record.status = "success"
        else:
            record.status = "timeout" if result.get("error_type") == "timeout" else "tool_error"
            record.error = result.get("error")

    def _fail(self, record: ToolRecord, status: str, error: str) -> None:
        record.status, record.error = status, error
