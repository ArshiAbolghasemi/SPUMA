"""`spn:` section: class-conditional SPFlow SPN hyper-parameters."""

from dataclasses import dataclass

from spuma.common.config import check_choice, check_range


@dataclass(frozen=True)
class SPNConfig:
    """`spn:` section."""

    backend: str = "spflow"
    model_type: str = "class_conditional"
    target: str = "incorrect"
    seed: int = 42
    leaf: str = "normal"
    min_instances_slice: int = 50
    min_features_slice: int = 2
    rdc_threshold: float = 0.3
    standardize: bool = True
    prior_smoothing: float = 1.0  # Laplace smoothing of P(Y)

    def __post_init__(self) -> None:
        check_choice("spn.backend", self.backend, ("spflow",))
        check_choice("spn.model_type", self.model_type, ("class_conditional",))
        check_choice("spn.target", self.target, ("incorrect",))
        check_choice("spn.leaf", self.leaf, ("normal", "laplace"))
        check_range("spn.min_instances_slice", self.min_instances_slice, ge=2)
        check_range("spn.min_features_slice", self.min_features_slice, ge=2)
        check_range("spn.rdc_threshold", self.rdc_threshold, gt=0.0, lt=1.0)
        check_range("spn.prior_smoothing", self.prior_smoothing, ge=0.0)
