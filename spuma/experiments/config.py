"""`project:` and `experiment:` sections."""

from dataclasses import dataclass
from pathlib import Path

from spuma.common.config import as_path, check_range


@dataclass(frozen=True)
class ProjectConfig:
    """`project:` section."""

    name: str = "spuma"
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """`experiment:` section."""

    name: str = "ma_qwen_math"
    output_dir: Path = Path("artifacts")
    resume: bool = True
    concurrent_problems: int = 1

    def __post_init__(self) -> None:
        as_path(self, "output_dir")
        check_range("experiment.concurrent_problems", self.concurrent_problems, ge=1)
