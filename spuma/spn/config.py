"""`spn:` section: SPFlow LearnSPN hyper-parameters and leaf distributions (shared by both SPN risk models)."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

LEAF_TYPES = ("normal", "laplace", "histogram")


@dataclass(frozen=True)
class HistogramConfig:
    """`spn.histogram:` piecewise-constant leaves (edges from the full training data, per feature)."""

    bins: int = 10
    strategy: str = "uniform"  # uniform: equal widths | quantile: equal training mass per bin
    smoothing: float = 1.0  # Laplace pseudo-count per cell (bins + 2 tails): no cell has probability 0

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
    leaf: str = "histogram"  # continuous feature leaves: normal | laplace | histogram
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    # LearnSPN: k-means row clustering (sum nodes), RDC variable partitioning (product nodes)
    min_instances_slice: int = 50
    min_features_slice: int = 2
    n_clusters: int = 2
    rdc_threshold: float = 0.3
    # joint SPN: never let a product node split a varying label Y into a block of its own (Y joins the
    # feature it depends on most); false = plain LearnSPN, which can make P(Y | Z) constant
    keep_label_dependence: bool = True
    standardize: bool = True
    # Laplace pseudo-count of the label: indicator leaves (joint) / P(Y) (class-conditional)
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
