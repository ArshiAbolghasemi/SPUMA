"""SPNs over the uncertainty features Z and the correctness label Y (Y = 1: the final answer is wrong).

`JointSPN` (risk model `spn`) is ONE sum-product network over (Z, Y), learned by LearnSPN over the columns
[Z, Y]. Y is a single binary variable represented by its two indicator leaves (`BinaryIndicatorLeaf`), so
the network polynomial gives every joint and marginal query exactly:

    log P(Z, Y=c) = log S(Z, I(+y) = [c = 1], I(−y) = [c = 0])
    log P(Z)      = log S(Z, I(+y) = 1, I(−y) = 1)                              (Y marginalized)
    U_SPUMA(Z)    = P(Y=1 | Z) = exp(log P(Z,1) − logsumexp_c log P(Z,c))     (predicted error risk)
    H(Y | Z)      = −p log p − (1−p) log(1−p)                                 (reported separately)

`ClassConditionalSPN` (risk model `spn_class_conditional`) is the earlier baseline: P(Y) · P(Z | Y) with one
LearnSPN density per class. Both use the same continuous leaves (`spn.leaf`) and log-space inference;
features given as NaN (or listed in `marginalize`) are integrated out exactly.
"""

import io
import logging
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from spflow.learn import learn_spn
from spflow.learn.learn_spn import partition_by_rdc
from spflow.meta.data.scope import Scope
from spflow.modules.products import Product
from spflow.modules.sums import Sum
from spflow.utils.rdc import rdc

from spuma.common.config import build_config, to_plain
from spuma.config import get_settings
from spuma.evaluation.metrics import binary_entropy
from spuma.spn.config import HistogramConfig, SPNConfig
from spuma.spn.leaves import CONTINUOUS_LEAVES, BinaryIndicatorLeaf, LeafSettings, histogram_edges, leaf_settings

logger = logging.getLogger(__name__)


def partition_with_constants(data: torch.Tensor, threshold: float) -> torch.Tensor:
    """RDC partitioning that first splits off constant columns (independent of everything).

    SPFlow's RDC raises on zero-variance columns, which real slices have (e.g. SE = 0 for every row, or a
    slice where every answer is correct). If RDC still fails numerically on a slice, the slice is fully
    factorized (and a warning logged).
    """
    d = data.shape[1]
    const = (data.std(0) == 0).cpu()
    ids = torch.zeros(d, dtype=torch.int)
    varying = torch.nonzero(~const).flatten()
    if len(varying) >= 2:
        try:
            ids[varying] = partition_by_rdc(data[:, varying], threshold=threshold).to(torch.int)
        except RuntimeError as e:
            logger.warning("RDC failed on a slice; factorizing it", extra={"error": str(e), "rows": len(data)})
            ids[varying] = torch.arange(1, len(varying) + 1, dtype=torch.int)
    elif len(varying) == 1:
        ids[varying] = 1
    for j in torch.nonzero(const).flatten():
        ids[j] = int(ids.max()) + 1
    return ids


def _label_column(data: torch.Tensor) -> int | None:
    """The slice's label column: the only column whose values are exactly {0, 1} (both present).

    LearnSPN hands the partitioner only the slice's data, not its scope; `JointSPN.fit` guarantees that no
    standardized feature column has this pattern, so the match is unambiguous.
    """
    cols = [j for j in range(data.shape[1]) if set(torch.unique(data[:, j]).tolist()) == {0.0, 1.0}]
    return cols[0] if len(cols) == 1 else None


def partition_keeping_label(data: torch.Tensor, threshold: float) -> torch.Tensor:
    """`partition_with_constants`, but a varying label is never split into a block of its own.

    A block {Y} under a product node would make Y independent of every feature in that branch, so P(Y | Z)
    could not depend on Z there. When RDC (a noisy test, and the label's dependence is often weak) proposes
    it, Y joins the block of the feature it is most dependent on instead. Merging blocks keeps the product
    decomposable, so the SPN stays valid; a label that is constant in the slice is still split off.
    """
    ids = partition_with_constants(data, threshold)
    j = _label_column(data)
    if j is None or int((ids == ids[j]).sum()) > 1:
        return ids
    others = [k for k in range(data.shape[1]) if k != j and float(data[:, k].std()) > 0]
    if not others:
        return ids
    scores = [float(rdc(data[:, j], data[:, k])) for k in others]
    ids[j] = ids[others[int(np.argmax(scores))]]
    return ids


def structure_stats(root: Any) -> dict[str, int]:
    modules = list(root.modules())
    return {
        "sums": sum(isinstance(m, Sum) for m in modules),
        "products": sum(isinstance(m, Product) for m in modules),
        "leaves": sum(hasattr(m, "_compute_parameter_estimates") for m in modules),
    }


class _SPN:
    """Shared parts: settings, standardization, leaf settings, LearnSPN call, checkpointing."""

    kind = ""

    def __init__(self) -> None:
        settings = get_settings()
        self._init(settings.spn, settings.features.order)

    def _init(self, config: SPNConfig, feature_names: list[str]) -> None:
        self.config = config
        self.feature_names = list(feature_names)
        self.mean = np.zeros(len(feature_names))
        self.std = np.ones(len(feature_names))

    def _standardize(self, Z: np.ndarray) -> torch.Tensor:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != len(self.feature_names):
            raise ValueError(f"expected shape (n, {len(self.feature_names)}), got {Z.shape}")
        if self.config.standardize:
            Z = (Z - self.mean) / self.std  # NaN stays NaN (marginalized)
        return torch.tensor(Z, dtype=torch.float64)

    def _features(self, Z: np.ndarray, marginalize: list[str] | None) -> torch.Tensor:
        data = self._standardize(Z).clone()
        for name in marginalize or []:
            data[:, self.feature_names.index(name)] = float("nan")
        return data

    def _prepare(self, Z: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, np.ndarray, LeafSettings]:
        Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y, dtype=int)
        counts = np.array([(y == 0).sum(), (y == 1).sum()])
        if (counts < 2).any():
            raise ValueError(f"each class needs at least 2 training rows, got counts {counts.tolist()}")
        if self.config.standardize:
            self.mean = Z.mean(0)
            self.std = np.where(Z.std(0) > 0, Z.std(0), 1.0)
        data = self._standardize(Z)
        edges = {}
        if self.config.leaf == "histogram":
            h: HistogramConfig = self.config.histogram
            edges = {
                j: torch.tensor(histogram_edges(data[:, j].numpy(), h.bins, h.strategy)) for j in range(Z.shape[1])
            }
        settings = LeafSettings(
            label_smoothing=self.config.prior_smoothing, bin_smoothing=self.config.histogram.smoothing, edges=edges
        )
        return data, counts, settings

    def _learn(self, data: torch.Tensor, leaf_modules: list[Any], partition: Any = partition_with_constants) -> Any:
        return learn_spn(
            data,
            leaf_modules=leaf_modules,
            min_instances_slice=self.config.min_instances_slice,
            min_features_slice=self.config.min_features_slice,
            clustering_args={"n_clusters": self.config.n_clusters},
            partitioning_method=partition,
            partitioning_args={"threshold": self.config.rdc_threshold},
        ).eval()

    def _feature_leaf(self, n_features: int) -> Any:
        return CONTINUOUS_LEAVES[self.config.leaf](scope=Scope(list(range(n_features))), out_channels=1)

    def log_joint(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        raise NotImplementedError

    def predict_error_proba(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        lj = self.log_joint(Z, marginalize)
        return np.exp(lj[:, 1] - np.logaddexp(lj[:, 0], lj[:, 1]))

    def posterior_entropy(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        return binary_entropy(self.predict_error_proba(Z, marginalize))

    def _state(self) -> dict[str, Any]:
        return {"kind": self.kind, "config": to_plain(self.config), "feature_names": self.feature_names}

    def _restore(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        state = {**self._state(), "mean": self.mean, "std": self.std}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | io.BytesIO) -> Self:
        # Our own checkpoint (pickled SPFlow modules); never load checkpoints from untrusted sources.
        state = torch.load(path, weights_only=False)
        if state.get("kind") != cls.kind:
            raise ValueError(
                f"checkpoint holds a {state.get('kind') or 'class_conditional (old format)'} SPN, not {cls.kind}"
            )
        model = cls.__new__(cls)  # the checkpoint's own config and features, not the current settings
        model._init(build_config(SPNConfig, state["config"]), state["feature_names"])
        model.mean, model.std = state["mean"], state["std"]
        model._restore(state)
        return model


class JointSPN(_SPN):
    """One SPN over (Z, Y); Y is variable index len(Z) with indicator leaves."""

    kind = "joint"

    def _init(self, config: SPNConfig, feature_names: list[str]) -> None:
        super()._init(config, feature_names)
        self.root: Any = None

    @property
    def label_index(self) -> int:
        return len(self.feature_names)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> Self:
        data, counts, settings = self._prepare(Z, y)
        if any(_label_column(data[:, [j]]) is not None for j in range(data.shape[1])):
            raise ValueError("a standardized feature takes exactly the values {0, 1}; it would look like the label")
        data = torch.cat([data, torch.as_tensor(np.asarray(y), dtype=torch.float64)[:, None]], dim=1)
        partition = partition_keeping_label if self.config.keep_label_dependence else partition_with_constants
        torch.manual_seed(self.config.seed)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with leaf_settings(settings):
                leaves = [self._feature_leaf(self.label_index), BinaryIndicatorLeaf(Scope([self.label_index]))]
                self.root = self._learn(data, leaves, partition)
        finally:
            torch.set_default_dtype(prev)
        stats = structure_stats(self.root)
        logger.info(
            "joint SPN trained",
            extra={
                "class_counts": counts.tolist(),
                "leaf": self.config.leaf,
                "log_prior": self.log_prior().tolist(),
                **stats,
            },
        )
        spread = float(np.ptp(self.predict_error_proba(np.asarray(Z, dtype=np.float64))))
        if spread < 1e-9:
            logger.warning("learned structure makes Y independent of Z: every prediction equals the prior")
        return self

    def _evidence(self, features: torch.Tensor, label: float) -> torch.Tensor:
        """[Z, Y] rows with Y = label (0, 1, or NaN = unobserved: both indicators 1)."""
        y = torch.full((len(features), 1), label, dtype=torch.float64)
        return torch.cat([features, y], dim=1)

    @torch.no_grad()
    def _log_s(self, data: torch.Tensor) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("SPN is not fitted")
        return self.root.log_likelihood(data).reshape(len(data), -1)[:, 0].numpy()

    def log_joint(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        """(n, 2) array of log P(Z, Y=c), c = 0, 1."""
        features = self._features(Z, marginalize)
        return np.stack([self._log_s(self._evidence(features, c)) for c in (0.0, 1.0)], axis=1)

    def log_marginal(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        """log P(Z) with Y unobserved (both indicators set to 1)."""
        return self._log_s(self._evidence(self._features(Z, marginalize), float("nan")))

    def log_prior(self) -> np.ndarray:
        """log P(Y=c) of the joint model (every feature marginalized)."""
        return self.log_joint(np.full((1, len(self.feature_names)), np.nan))[0]

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.kind,
            "log_prior": self.log_prior().tolist(),
            "leaf": self.config.leaf,
            "features": self.feature_names,
            **structure_stats(self.root),
        }

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "root": self.root}

    def _restore(self, state: dict[str, Any]) -> None:
        self.root = state["root"]


class ClassConditionalSPN(_SPN):
    """Baseline: P(Z, Y=c) = P(Y=c) · SPN_c(Z), one LearnSPN density per class."""

    kind = "class_conditional"

    def _init(self, config: SPNConfig, feature_names: list[str]) -> None:
        super()._init(config, feature_names)
        self.spns: dict[int, Any] = {}
        self.log_prior_ = np.zeros(2)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> Self:
        data, counts, settings = self._prepare(Z, y)
        y = np.asarray(y, dtype=int)
        a = self.config.prior_smoothing
        self.log_prior_ = np.log((counts + a) / (counts.sum() + 2 * a))
        torch.manual_seed(self.config.seed)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            with leaf_settings(settings):
                for c in (0, 1):
                    self.spns[c] = self._learn(data[torch.from_numpy(y == c)], [self._feature_leaf(data.shape[1])])
        finally:
            torch.set_default_dtype(prev)
        logger.info(
            "class-conditional SPN trained",
            extra={"class_counts": counts.tolist(), "log_prior": self.log_prior_.tolist(), "leaf": self.config.leaf},
        )
        return self

    @torch.no_grad()
    def log_joint(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        """(n, 2) array of log P(Z, Y=c)."""
        if not self.spns:
            raise RuntimeError("SPN is not fitted")
        data = self._features(Z, marginalize)
        cols = [self.spns[c].log_likelihood(data).reshape(len(data), -1)[:, 0].numpy() for c in (0, 1)]
        return np.stack(cols, axis=1) + self.log_prior_

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.kind,
            "log_prior": self.log_prior_.tolist(),
            "leaf": self.config.leaf,
            "features": self.feature_names,
        }

    def _state(self) -> dict[str, Any]:
        return {**super()._state(), "spns": self.spns, "log_prior": self.log_prior_}

    def _restore(self, state: dict[str, Any]) -> None:
        self.spns, self.log_prior_ = state["spns"], state["log_prior"]
