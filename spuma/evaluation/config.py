"""`evaluation:` section."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

# Names registered in spuma.evaluation.risk.RISK_MODELS; extend both when adding a risk model.
RISK_MODEL_NAMES = ("spn", "spn_class_conditional")


@dataclass(frozen=True)
class BootstrapConfig:
    """`evaluation.bootstrap:` section."""

    enabled: bool = True
    iterations: int = 1000
    confidence: float = 0.95
    cluster_by: str = "problem_id"

    def __post_init__(self) -> None:
        check_range("evaluation.bootstrap.iterations", self.iterations, gt=0)
        check_range("evaluation.bootstrap.confidence", self.confidence, gt=0.0, lt=1.0)
        check_choice("evaluation.bootstrap.cluster_by", self.cluster_by, ("problem_id",))


@dataclass(frozen=True)
class EvaluationConfig:
    """`evaluation:` section."""

    risk_models: list[str] = field(default_factory=lambda: ["spn"])
    metrics: list[str] = field(default_factory=lambda: ["auroc", "auarc"])
    ece_bins: int = 15
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def __post_init__(self) -> None:
        if not self.risk_models:
            raise ValueError("evaluation.risk_models must name at least one model")
        for name in self.risk_models:
            check_choice("evaluation.risk_models[]", name, RISK_MODEL_NAMES)
        check_range("evaluation.ece_bins", self.ece_bins, gt=0)
