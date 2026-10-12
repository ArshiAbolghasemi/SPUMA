"""`dataset:` section."""

from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig


class DatasetConfig(StrictConfig):
    """`dataset:` section."""

    name: Literal["math"] = "math"
    source: Literal["huggingface"] = "huggingface"
    hf_path: str = "EleutherAI/hendrycks_math"
    subsets: list[str] = Field(
        default_factory=lambda: [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ]
    )
    seed: int = 42
    train_samples: int = Field(800, ge=0)  # drawn from the official train split
    validation_samples: int = Field(200, ge=0)  # drawn from the official train split, disjoint from train
    test_samples: int = Field(500, ge=0)  # drawn from the official test split
