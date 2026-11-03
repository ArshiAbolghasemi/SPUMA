"""`spn:` section: SPFlow LearnSPN hyper-parameters and leaf distributions (shared by both SPN risk models)."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

LEAF_TYPES = ("normal", "laplace", "histogram")


@dataclass(frozen=True)
class HistogramConfig:
    """`spn.histogram:` piecewise-constant leaves (edges from the full training data, per feature)."""

    bins: int = 10
    strategy: str = "uniform"
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        check_range("spn.histogram.bins", self.bins, ge=1)
        check_choice("spn.histogram.strategy", self.strategy, ("uniform", "quantile"))
        check_range("spn.histogram.smoothing", self.smoothing, gt=0.0)


@dataclass(frozen=True)
class SPNConfig:
    """`spn:` section."""

    backend: str = "spflow"
    target: str = "incorrect"
    seed: int = 42
    leaf: str = "histogram"
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    min_instances_slice: int = 50
    min_features_slice: int = 2
    n_clusters: int = 2
    rdc_threshold: float = 0.3
    keep_label_dependence: bool = True
    standardize: bool = True
    prior_smoothing: float = 1.0

    def __post_init__(self) -> None:
        check_choice("spn.backend", self.backend, ("spflow",))
        check_choice("spn.target", self.target, ("incorrect",))
        check_choice("spn.leaf", self.leaf, LEAF_TYPES)
        check_range("spn.min_instances_slice", self.min_instances_slice, ge=2)
        check_range("spn.min_features_slice", self.min_features_slice, ge=2)
        check_range("spn.n_clusters", self.n_clusters, ge=2)
        check_range("spn.rdc_threshold", self.rdc_threshold, gt=0.0, lt=1.0)
        check_range("spn.prior_smoothing", self.prior_smoothing, ge=0.0)


@dataclass(frozen=True)
class ClassConditionalSPNConfig:
    """`class_conditional_spn:` P(Z | Y=c) as one LearnSPN density per class, combined with the empirical
    training prior by Bayes' rule. Every other LearnSPN/leaf setting and the preprocessing come from `spn:`.

    Each list is a validation grid: one candidate per combination of leaf, `rdc_threshold` and
    `min_instances_slice`, crossed with that leaf's regularizer (`variance_floor` for normal leaves,
    `histogram_smoothing` for histogram leaves, none for laplace). Every candidate is fitted on the training
    split; the one with the lowest validation Brier score is kept.
    """

    enabled: bool = False
    leaves: list[str] = field(default_factory=lambda: ["normal", "histogram"])
    rdc_threshold: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    min_instances_slice: list[int] = field(default_factory=lambda: [25, 50, 100])
    # normal leaves: sigma^2 >= floor, on standardized features (unit training variance)
    variance_floor: list[float] = field(default_factory=lambda: [0.01, 0.1])
    # histogram leaves: Laplace pseudo-count per cell
    histogram_smoothing: list[float] = field(default_factory=lambda: [1.0, 5.0])

    def __post_init__(self) -> None:
        for name in ("leaves", "rdc_threshold", "min_instances_slice", "variance_floor", "histogram_smoothing"):
            if not getattr(self, name):
                raise ValueError(f"class_conditional_spn.{name} must not be empty")
        for v in self.leaves:
            check_choice("class_conditional_spn.leaves[]", v, LEAF_TYPES)
        for v in self.rdc_threshold:
            check_range("class_conditional_spn.rdc_threshold[]", v, gt=0.0, lt=1.0)
        for v in self.min_instances_slice:
            check_range("class_conditional_spn.min_instances_slice[]", v, ge=2)
        for v in self.variance_floor:
            check_range("class_conditional_spn.variance_floor[]", v, gt=0.0)
        for v in self.histogram_smoothing:
            check_range("class_conditional_spn.histogram_smoothing[]", v, gt=0.0)
