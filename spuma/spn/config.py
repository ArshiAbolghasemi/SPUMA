"""`spn:` section: class-conditional SPFlow SPN hyper-parameters."""

from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig


class SPNConfig(StrictConfig):
    """`spn:` section."""

    backend: Literal["spflow"] = "spflow"
    model_type: Literal["class_conditional"] = "class_conditional"
    target: Literal["incorrect"] = "incorrect"
    seed: int = 42
    leaf: Literal["normal", "laplace"] = "normal"
    min_instances_slice: int = Field(50, ge=2)
    min_features_slice: int = Field(2, ge=2)
    rdc_threshold: float = Field(0.3, gt=0.0, lt=1.0)
    standardize: bool = True
    prior_smoothing: float = Field(1.0, ge=0.0)  # Laplace smoothing of P(Y)
