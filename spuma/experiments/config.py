"""`project:` and `experiment:` sections."""

from dataclasses import dataclass, field
from pathlib import Path

from spuma.common.config import as_path, check_choice


@dataclass(frozen=True)
class ProjectConfig:
    """`project:` section."""

    name: str = "spuma"
    seed: int = 42


@dataclass(frozen=True)
class ToolVariant:
    """One `experiment.variants` entry: per-agent `allowed_tools` overrides; agents not listed keep theirs."""

    allowed_tools: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """`experiment:` section."""

    name: str = "ma_qwen_math"
    output_dir: Path = Path("artifacts")
    resume: bool = True
    variants: dict[str, ToolVariant] = field(default_factory=lambda: {"default": ToolVariant()})
    # within_variant: train and test the risk models inside each variant.
    # cross_variant: additionally evaluate every variant's models on every other variant's test set.
    spn_protocol: str = "within_variant"

    def __post_init__(self) -> None:
        as_path(self, "output_dir")
        check_choice("experiment.spn_protocol", self.spn_protocol, ("within_variant", "cross_variant"))
        if not self.variants:
            raise ValueError("experiment.variants must define at least one variant")
