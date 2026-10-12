"""`project:` and `experiment:` sections."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig


class ProjectConfig(StrictConfig):
    """`project:` section."""

    name: str = "spuma"
    seed: int = 42


class ToolVariant(StrictConfig):
    """Per-agent `allowed_tools` overrides for one experimental condition; agents not listed keep theirs."""

    allowed_tools: dict[str, list[str]] = Field(default_factory=dict)


class ExperimentConfig(StrictConfig):
    """`experiment:` section."""

    name: str = "ma_qwen_math"
    output_dir: Path = Path("artifacts")
    resume: bool = True
    variants: dict[str, ToolVariant] = Field(default_factory=lambda: {"default": ToolVariant()})
    # within_variant: train and test the risk models inside each variant.
    # cross_variant: additionally evaluate every variant's models on every other variant's test set.
    spn_protocol: Literal["within_variant", "cross_variant"] = "within_variant"
