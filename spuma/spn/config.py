"""`spn:` section: SPFlow LearnSPN hyper-parameters and leaf distributions (shared by both SPN risk models).

`leaf`, `rdc_threshold`, `min_instances_slice`, `variance_floor` and `histogram.smoothing` take a value or a list.
Lists span a validation grid (`SPNConfig.candidates`): one candidate per leaf × RDC threshold × slice size, crossed
with that leaf's regularizer (`variance_floor` for normal leaves, `histogram.smoothing` for histogram leaves, none
for laplace). Every candidate is fitted on the training split and the lowest validation Brier score is kept, for
the joint and the class-conditional SPN alike.
"""

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

from spuma.common.config import check_choice, check_range

LEAF_TYPES = ("normal", "laplace", "histogram")


def values(v: Any) -> list[Any]:
    """A setting that is a value or a list of values, as a list."""
    return list(v) if isinstance(v, list | tuple) else [v]


def _check_values(key: str, v: Any, **bounds: float) -> None:
    if not values(v):
        raise ValueError(f"{key} must not be an empty list")
    for x in values(v):
        check_range(key, x, **bounds)


@dataclass(frozen=True)
class HistogramConfig:
    """`spn.histogram:` piecewise-constant leaves (edges from the full training data, per feature)."""

    bins: int = 10
    strategy: str = "uniform"
    smoothing: float | list[float] = 1.0  # Laplace pseudo-count per cell

    def __post_init__(self) -> None:
        check_range("spn.histogram.bins", self.bins, ge=1)
        check_choice("spn.histogram.strategy", self.strategy, ("uniform", "quantile"))
        _check_values("spn.histogram.smoothing", self.smoothing, gt=0.0)


@dataclass(frozen=True)
class SPNConfig:
    """`spn:` section."""

    backend: str = "spflow"
    target: str = "incorrect"
    seed: int = 42
    leaf: str | list[str] = "histogram"
    histogram: HistogramConfig = field(default_factory=HistogramConfig)
    min_instances_slice: int | list[int] = 50
    min_features_slice: int = 2
    n_clusters: int = 2
    rdc_threshold: float | list[float] = 0.3
    # normal leaves: sigma^2 >= floor, on standardized features (unit training variance); 0 = no floor
    variance_floor: float | list[float] = 0.0
    keep_label_dependence: bool = True
    standardize: bool = True
    prior_smoothing: float = 1.0

    def __post_init__(self) -> None:
        check_choice("spn.backend", self.backend, ("spflow",))
        check_choice("spn.target", self.target, ("incorrect",))
        if not values(self.leaf):
            raise ValueError("spn.leaf must not be an empty list")
        for leaf in values(self.leaf):
            check_choice("spn.leaf", leaf, LEAF_TYPES)
        _check_values("spn.min_instances_slice", self.min_instances_slice, ge=2)
        check_range("spn.min_features_slice", self.min_features_slice, ge=2)
        check_range("spn.n_clusters", self.n_clusters, ge=2)
        _check_values("spn.rdc_threshold", self.rdc_threshold, gt=0.0, lt=1.0)
        _check_values("spn.variance_floor", self.variance_floor, ge=0.0)
        check_range("spn.prior_smoothing", self.prior_smoothing, ge=0.0)

    def candidates(self) -> list["SPNConfig"]:
        """Every grid point as an all-scalar config (one, itself, when nothing is a list); settings that do not
        apply to a candidate's leaf are set to their default."""
        out = []
        for leaf, rdc, mis in itertools.product(
            values(self.leaf), values(self.rdc_threshold), values(self.min_instances_slice)
        ):
            base = dataclasses.replace(
                self,
                leaf=leaf,
                rdc_threshold=rdc,
                min_instances_slice=mis,
                variance_floor=0.0,
                histogram=dataclasses.replace(self.histogram, smoothing=values(self.histogram.smoothing)[0]),
            )
            if leaf == "normal":
                out += [dataclasses.replace(base, variance_floor=v) for v in values(self.variance_floor)]
            elif leaf == "histogram":
                out += [
                    dataclasses.replace(base, histogram=dataclasses.replace(self.histogram, smoothing=a))
                    for a in values(self.histogram.smoothing)
                ]
            else:
                out.append(base)
        return out


@dataclass(frozen=True)
class ClassConditionalSPNConfig:
    """`class_conditional_spn:` also train P(Z | Y=c) as one LearnSPN density per class, combined with the
    empirical training prior by Bayes' rule. Every setting, grid included, and the preprocessing come from `spn:`."""

    enabled: bool = False
