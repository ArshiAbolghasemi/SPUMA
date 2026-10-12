"""`agents:` and `orchestration:` sections."""

from typing import Literal

from pydantic import Field, field_validator

from spuma.common.config import StrictConfig
from spuma.models.config import GenerationConfig


class AgentConfig(StrictConfig):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    model: str = "default"
    role: str
    enabled: bool = True
    system_prompt: str
    # Rendered with {problem} and, for later agents, {previous} (earlier agents' final messages).
    user_template: str = "{problem}"
    allowed_tools: list[str] = Field(default_factory=list)
    generation: GenerationConfig | None = None  # None: the model's generation settings
    max_steps: int | None = Field(None, gt=0)  # None: orchestration.max_steps_per_agent
    max_tool_calls: int | None = Field(None, ge=0)  # None: orchestration.max_tool_calls_per_agent

    @field_validator("allowed_tools")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError(f"allowed_tools has duplicates: {v}")
        return v


class AgentsConfig(StrictConfig):
    """`agents:` section."""

    workflow: Literal["sequential"] = "sequential"
    items: list[AgentConfig]

    @property
    def enabled(self) -> list[AgentConfig]:
        return [a for a in self.items if a.enabled]


class OrchestrationConfig(StrictConfig):
    """`orchestration:` section."""

    framework: Literal["langgraph"] = "langgraph"
    max_steps_per_agent: int = Field(8, gt=0)
    max_tool_calls_per_agent: int = Field(5, ge=0)
