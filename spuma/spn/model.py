"""Class-conditional SPN over (Z, Y):  P(Z, Y) = P(Y) · P(Z | Y), one SPFlow LearnSPN density per class.

    log P(Z, Y=c) = log P(Y=c) + log SPN_c(Z)
    U_SPUMA(Z)    = P(Y=1 | Z) = exp(log P(Z,1) - logsumexp_c log P(Z,c))      (predicted error risk)
    H(Y | Z)      = -p log p - (1-p) log(1-p)                                  (reported separately)

Everything stays in log space. Features given as NaN (or listed in `marginalize`) are integrated
out exactly by SPFlow, so the posterior is well defined with missing features.
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
from spflow.modules.leaves import Laplace, Normal

from spuma.config import get_settings
from spuma.evaluation.metrics import binary_entropy
from spuma.spn.config import SPNConfig

logger = logging.getLogger(__name__)
LEAVES = {"normal": Normal, "laplace": Laplace}


def partition_with_constants(data: torch.Tensor, threshold: float) -> torch.Tensor:
    """RDC partitioning that first splits off constant columns (independent of everything).

    SPFlow's RDC raises on zero-variance columns, which real slices have (e.g. SE = 0 for every row).
    If RDC still fails numerically on a slice, the slice is fully factorized (and a warning logged).
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


class ClassConditionalSPN:
    """Built from the `spn:` settings over the `features.order` features; `load` restores a checkpoint as saved."""

    def __init__(self) -> None:
        settings = get_settings()
        self._init(settings.spn, settings.features.order)

    def _init(self, config: SPNConfig, feature_names: list[str]) -> None:
        self.config = config
        self.feature_names = list(feature_names)
        self.spns: dict[int, Any] = {}
        self.log_prior = np.zeros(2)
        self.mean = np.zeros(len(feature_names))
        self.std = np.ones(len(feature_names))

    def _standardize(self, Z: np.ndarray) -> torch.Tensor:
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != len(self.feature_names):
            raise ValueError(f"expected shape (n, {len(self.feature_names)}), got {Z.shape}")
        if self.config.standardize:
            Z = (Z - self.mean) / self.std  # NaN stays NaN (marginalized)
        return torch.tensor(Z, dtype=torch.float64)

    def fit(self, Z: np.ndarray, y: np.ndarray) -> Self:
        Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y, dtype=int)
        counts = np.array([(y == 0).sum(), (y == 1).sum()])
        if (counts < 2).any():
            raise ValueError(f"each class needs at least 2 training rows, got counts {counts.tolist()}")
        if self.config.standardize:
            self.mean = Z.mean(0)
            self.std = np.where(Z.std(0) > 0, Z.std(0), 1.0)
        a = self.config.prior_smoothing
        self.log_prior = np.log((counts + a) / (counts.sum() + 2 * a))
        torch.manual_seed(self.config.seed)
        data = self._standardize(Z)
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            for c in (0, 1):
                leaf = LEAVES[self.config.leaf](scope=Scope(list(range(Z.shape[1]))), out_channels=1)
                self.spns[c] = learn_spn(
                    data[torch.from_numpy(y == c)],
                    leaf_modules=leaf,
                    min_instances_slice=self.config.min_instances_slice,
                    min_features_slice=self.config.min_features_slice,
                    partitioning_method=partition_with_constants,
                    partitioning_args={"threshold": self.config.rdc_threshold},
                ).eval()
        finally:
            torch.set_default_dtype(prev)
        logger.info(
            "SPN trained",
            extra={"class_counts": counts.tolist(), "log_prior": self.log_prior.tolist(), "leaf": self.config.leaf},
        )
        return self

    @torch.no_grad()
    def log_joint(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        """(n, 2) array of log P(Z, Y=c)."""
        if not self.spns:
            raise RuntimeError("SPN is not fitted")
        data = self._standardize(Z).clone()
        for name in marginalize or []:
            data[:, self.feature_names.index(name)] = float("nan")
        cols = [self.spns[c].log_likelihood(data).reshape(len(data), -1)[:, 0].numpy() for c in (0, 1)]
        return np.stack(cols, axis=1) + self.log_prior

    def predict_error_proba(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        lj = self.log_joint(Z, marginalize)
        return np.exp(lj[:, 1] - np.logaddexp(lj[:, 0], lj[:, 1]))

    def posterior_entropy(self, Z: np.ndarray, marginalize: list[str] | None = None) -> np.ndarray:
        return binary_entropy(self.predict_error_proba(Z, marginalize))

    def summary(self) -> dict[str, Any]:
        return {"log_prior": self.log_prior.tolist(), "leaf": self.config.leaf, "features": self.feature_names}

    def save(self, path: Path) -> None:
        state = {
            "config": self.config.model_dump(),
            "feature_names": self.feature_names,
            "spns": self.spns,
            "log_prior": self.log_prior,
            "mean": self.mean,
            "std": self.std,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load(cls, path: Path | io.BytesIO) -> ClassConditionalSPN:
        # Our own checkpoint (pickled SPFlow modules); never load checkpoints from untrusted sources.
        state = torch.load(path, weights_only=False)
        model = cls.__new__(cls)  # the checkpoint's own config and features, not the current settings
        model._init(SPNConfig(**state["config"]), state["feature_names"])
        model.spns, model.log_prior = state["spns"], state["log_prior"]
        model.mean, model.std = state["mean"], state["std"]
        return model
