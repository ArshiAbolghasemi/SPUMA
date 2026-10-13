"""`agents:` and `orchestration:` sections."""

import re
from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range
from spuma.models.config import GenerationConfig


@dataclass(frozen=True)
class AgentConfig:
    """One `agents.items` entry."""

    id: str
    role: str
    system_prompt: str
    model: str = "default"
    enabled: bool = True
    # Rendered with {problem} and, for later agents, {previous} (earlier agents' final messages).
    user_template: str = "{problem}"
    allowed_tools: list[str] = field(default_factory=list)
    generation: GenerationConfig | None = None  # None: the model's generation settings
    max_steps: int | None = None  # None: orchestration.max_steps_per_agent
    max_tool_calls: int | None = None  # None: orchestration.max_tool_calls_per_agent

    def __post_init__(self) -> None:
        key = f"agents.items[{self.id}]"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError(f"{key}.id must match [a-z][a-z0-9_]*, got {self.id!r}")
        object.__setattr__(self, "allowed_tools", list(self.allowed_tools))
        if len(self.allowed_tools) != len(set(self.allowed_tools)):
            raise ValueError(f"{key}.allowed_tools has duplicates: {self.allowed_tools}")
        check_range(f"{key}.max_steps", self.max_steps, gt=0)
        check_range(f"{key}.max_tool_calls", self.max_tool_calls, ge=0)


@dataclass(frozen=True)
class AgentsConfig:
    """`agents:` section."""

    items: list[AgentConfig]
    workflow: str = "sequential"

    def __post_init__(self) -> None:
        check_choice("agents.workflow", self.workflow, ("sequential",))

    @property
    def enabled(self) -> list[AgentConfig]:
        return [a for a in self.items if a.enabled]


@dataclass(frozen=True)
class OrchestrationConfig:
    """`orchestration:` section."""

    framework: str = "langgraph"
    max_steps_per_agent: int = 8
    max_tool_calls_per_agent: int = 5

    def __post_init__(self) -> None:
        check_choice("orchestration.framework", self.framework, ("langgraph",))
        check_range("orchestration.max_steps_per_agent", self.max_steps_per_agent, gt=0)
        check_range("orchestration.max_tool_calls_per_agent", self.max_tool_calls_per_agent, ge=0)
