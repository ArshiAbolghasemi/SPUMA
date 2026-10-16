"""`dataset:` section."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


@dataclass(frozen=True)
class DatasetConfig:
    """`dataset:` section."""

    name: str = "math"
    source: str = "huggingface"
    hf_path: str = "EleutherAI/hendrycks_math"
    subsets: list[str] = field(default_factory=lambda: list(MATH_SUBJECTS))
    seed: int = 42
    train_samples: int = 800
    validation_samples: int = 200
    test_samples: int = 500

    def __post_init__(self) -> None:
        check_choice("dataset.name", self.name, ("math",))
        check_choice("dataset.source", self.source, ("huggingface",))
        for k in ("train_samples", "validation_samples", "test_samples"):
            check_range(f"dataset.{k}", getattr(self, k), ge=0)
