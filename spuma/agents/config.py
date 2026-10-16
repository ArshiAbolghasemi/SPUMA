"""`agents:` section."""

import re
from dataclasses import dataclass

from spuma.common.config import check_choice
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

    def __post_init__(self) -> None:
        check_choice("agents.workflow", self.workflow, ("sequential",))

    @property
    def enabled(self) -> list[AgentConfig]:
        return [a for a in self.items if a.enabled]
