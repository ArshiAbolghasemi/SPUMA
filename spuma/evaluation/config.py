"""`evaluation:` section."""

from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig

# Names registered in spuma.evaluation.risk.RISK_MODELS; extend both when adding a risk model.
RiskModelName = Literal["spn"]


class BootstrapConfig(StrictConfig):
    enabled: bool = True
    iterations: int = Field(1000, gt=0)
    confidence: float = Field(0.95, gt=0.0, lt=1.0)
    cluster_by: Literal["problem_id"] = "problem_id"


class EvaluationConfig(StrictConfig):
    """`evaluation:` section."""

    risk_models: list[RiskModelName] = Field(default_factory=lambda: ["spn"], min_length=1)
    metrics: list[str] = Field(default_factory=lambda: ["auroc", "auarc"])
    ece_bins: int = Field(15, gt=0)
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
