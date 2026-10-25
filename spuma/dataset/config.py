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
    revision: str | None = None
    # one file (morehopqa) or {official split: file}; `hf://` URLs or local paths (`source: local`)
    data_files: str | dict[str, str] | None = None
    subsets: list[str] = field(default_factory=lambda: list(MATH_SUBJECTS))
    seed: int = 42
    train_samples: int = 800
    validation_samples: int = 200
    test_samples: int | None = 500  # None: every remaining problem of the test source (medhopqa: all labeled)
    variant: str | None = None  # musique: ans | full
    # {train|validation|test: official split}; empty: the dataset's documented default
    split_sources: dict[str, str] = field(default_factory=dict)
    # medhopqa: {id|question|answer|synonyms: source column}; empty: detected from the file and logged
    fields: dict[str, str] = field(default_factory=dict)
    labeled_only: bool = True  # medhopqa: keep only questions that have a gold answer

    def __post_init__(self) -> None:
        check_choice("dataset.name", self.name, ("math", "morehopqa", "mmlu", "musique", "medhopqa"))
        check_choice("dataset.source", self.source, ("huggingface", "local"))
        if self.name == "musique":
            check_choice("dataset.variant", self.variant, ("ans", "full"))
        for target in self.split_sources:
            check_choice("dataset.split_sources", target, ("train", "validation", "test"))
        for k in ("train_samples", "validation_samples", "test_samples"):
            check_range(f"dataset.{k}", getattr(self, k), ge=0)
