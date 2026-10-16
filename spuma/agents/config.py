"""`agents:` section."""

import re
from dataclasses import dataclass

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
    user_template: str = "{problem}"
    generation: GenerationConfig | None = None

    def __post_init__(self) -> None:
        key = f"agents.items[{self.id}]"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError(f"{key}.id must match [a-z][a-z0-9_]*, got {self.id!r}")


@dataclass(frozen=True)
class AgentsConfig:
    """`agents:` section."""

    items: list[AgentConfig]
    workflow: str = "sequential"
    max_rounds: int = 10
    completion_token: str = "TASK_DONE"

    def __post_init__(self) -> None:
        check_choice("agents.workflow", self.workflow, ("sequential", "feedback"))
        check_range("agents.max_rounds", self.max_rounds, gt=0)
        if not self.completion_token.strip():
            raise ValueError("agents.completion_token must not be empty")
        if self.workflow == "feedback" and len(self.enabled) != 2:
            raise ValueError(f"agents.workflow feedback needs exactly 2 enabled agents, got {len(self.enabled)}")

    @property
    def enabled(self) -> list[AgentConfig]:
        return [a for a in self.items if a.enabled]
