"""`agents:` section."""

import re
from dataclasses import dataclass, field
from pathlib import Path

from spuma.common.config import check_choice, check_range
from spuma.models.config import GenerationConfig

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


def compose_prompt(parts: list[str]) -> str:
    """The fragments `config/prompts/<part>.txt`, in order, separated by a blank line."""
    texts = []
    for part in parts:
        path = PROMPTS_DIR / f"{part}.txt"
        if not path.is_file():
            raise ValueError(f"prompt fragment {part!r} not found ({path})")
        texts.append(path.read_text().strip())
    return "\n\n".join(texts)


@dataclass(frozen=True)
class AgentConfig:
    """One `agents.items` entry. The system prompt is `system_prompt`, or composed from `prompt`: fragment names
    under config/prompts/ (general rules + dataset instructions + role + verification rules + output schema)."""

    id: str
    role: str
    system_prompt: str = ""
    model: str = "default"
    enabled: bool = True
    user_template: str = "{problem}"
    generation: GenerationConfig | None = None
    # named user templates, one per call kind (the star coordinator's phases: STAR_PHASES)
    templates: dict[str, str] = field(default_factory=dict)
    prompt: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        key = f"agents.items[{self.id}]"
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.id):
            raise ValueError(f"{key}.id must match [a-z][a-z0-9_]*, got {self.id!r}")
        if self.prompt:
            composed = compose_prompt(self.prompt)
            if self.system_prompt and self.system_prompt != composed:
                raise ValueError(f"{key}: give system_prompt or prompt (fragments), not both")
            object.__setattr__(self, "system_prompt", composed)
        if not self.system_prompt.strip():
            raise ValueError(f"{key}: system_prompt or prompt (fragments) is required")


STAR_PHASES = ("delegation", "verification_request", "final_synthesis")


@dataclass(frozen=True)
class AgentsConfig:
    """`agents:` section."""

    items: list[AgentConfig]
    workflow: str = "sequential"
    max_rounds: int = 10
    completion_token: str = "TASK_DONE"

    def __post_init__(self) -> None:
        check_choice("agents.workflow", self.workflow, ("sequential", "feedback", "star"))
        check_range("agents.max_rounds", self.max_rounds, gt=0)
        if not self.completion_token.strip():
            raise ValueError("agents.completion_token must not be empty")
        if self.workflow == "feedback" and len(self.enabled) != 2:
            raise ValueError(f"agents.workflow feedback needs exactly 2 enabled agents, got {len(self.enabled)}")
        if self.workflow == "sequential" and len(self.enabled) != 3:
            raise ValueError(
                f"agents.workflow sequential needs exactly 3 enabled agents (solver, verifier, finalizer), "
                f"got {len(self.enabled)}"
            )
        if self.workflow == "star":
            if len(self.enabled) != 3:
                raise ValueError(
                    f"agents.workflow star needs exactly 3 enabled agents (coordinator, analyst, verifier), "
                    f"got {len(self.enabled)}"
                )
            missing = [p for p in STAR_PHASES if p not in self.enabled[0].templates]
            if missing:
                raise ValueError(f"agents.items[{self.enabled[0].id}].templates (star coordinator) lacks {missing}")

    @property
    def enabled(self) -> list[AgentConfig]:
        return [a for a in self.items if a.enabled]
