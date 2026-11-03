"""Custom SPFlow leaves for SPUMA's joint SPN: a binary indicator leaf for Y and a smoothed histogram leaf.

LearnSPN creates leaves as `LeafClass(scope=..., out_channels=...)` and fits them by maximum likelihood,
so per-fit settings the constructor cannot receive (bin edges, smoothing) come from `leaf_settings`,
active while the structure is learned; each leaf copies them into its own buffers, so fitted leaves
(and checkpoints) are self-contained.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from spflow.meta.data.scope import Scope
from spflow.modules.leaves import Laplace, Normal
from spflow.modules.leaves.leaf import LeafModule
from torch import Tensor, nn


@dataclass(frozen=True)
class LeafSettings:
    label_smoothing: float
    bin_smoothing: float
    edges: dict[int, Tensor]


_SETTINGS: ContextVar[LeafSettings | None] = ContextVar("spuma_spn_leaf_settings", default=None)


@contextmanager
def leaf_settings(settings: LeafSettings) -> Iterator[None]:
    token = _SETTINGS.set(settings)
    try:
        yield
    finally:
        _SETTINGS.reset(token)


def _active() -> LeafSettings:
    settings = _SETTINGS.get()
    if settings is None:
        raise RuntimeError("SPUMA leaves are created inside `leaf_settings(...)` (or from explicit parameters)")
    return settings


class BinaryIndicatorLeaf(LeafModule):
    """One binary variable Y as a sum node over its two indicator leaves (network-polynomial form):

        S(y) = θ · I(+y) + (1 − θ) · I(−y),   I(+y) = [Y = 1],  I(−y) = [Y = 0]

    Evidence Y = 1 sets (I(+y), I(−y)) = (1, 0), Y = 0 sets (0, 1), and Y unobserved (NaN) sets both to 1,
    which marginalizes Y exactly: S = θ + (1 − θ) = 1. θ is the (Laplace-smoothed) weighted frequency of
    Y = 1 among the rows that reach this leaf.
    """

    def __init__(
        self,
        scope: Scope | int | list[int],
        out_channels: int = 1,
        num_repetitions: int = 1,
        log_weights: Tensor | None = None,
        smoothing: float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if log_weights is not None:
            out_channels, num_repetitions = log_weights.shape[1], log_weights.shape[2]
        super().__init__(scope, out_channels, num_repetitions, validate_args=validate_args)
        self.smoothing = _active().label_smoothing if smoothing is None else float(smoothing)
        if log_weights is None:
            log_weights = torch.full((*self._event_shape, 2), math.log(0.5))
        self.log_weights = nn.Parameter(torch.as_tensor(log_weights, dtype=torch.float64))

    @staticmethod
    def indicators(y: Tensor) -> tuple[Tensor, Tensor]:
        """(I(+y), I(−y)) for evidence y in {0, 1, NaN}."""
        if not torch.all(torch.isnan(y) | (y == 0) | (y == 1)):
            raise ValueError("binary indicator evidence must be 0, 1 or NaN (unobserved)")
        missing = torch.isnan(y)
        return ((y == 1) | missing).to(y.dtype), ((y == 0) | missing).to(y.dtype)

    def log_likelihood(self, data: Tensor, cache: object = None) -> Tensor:
        y = data[:, self.scope.query].to(self.log_weights.dtype)
        pos, neg = self.indicators(y)
        log_ind = torch.stack([neg, pos], dim=-1).log()
        return torch.logsumexp(self.log_weights.unsqueeze(0) + log_ind[:, :, None, None, :], dim=-1)

    @property
    def _torch_distribution_class(self) -> None:
        return None

    @property
    def _supported_value(self) -> float:
        return 0.0

    def params(self) -> dict[str, Tensor]:
        return {"log_weights": self.log_weights}

    def _compute_parameter_estimates(self, data: Tensor, weights: Tensor, bias_correction: bool) -> dict[str, Tensor]:
        y = data[..., 0, 0]
        if torch.isnan(y).any():
            raise ValueError("the correctness label must be observed for every training row")
        self.indicators(y)
        w = weights
        positive = (w * y[:, :, None, None]).sum(0)
        total = w.sum(0)
        theta = (positive + self.smoothing) / (total + 2 * self.smoothing)
        return {"log_weights": torch.stack([torch.log1p(-theta), torch.log(theta)], dim=-1)}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        self.log_weights.data = params_dict["log_weights"].to(self.log_weights.dtype)

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: object = None) -> LeafModule | None:
        keep = [i for i, q in enumerate(self.scope.query) if q not in marg_rvs]
        if not keep:
            return None
        return BinaryIndicatorLeaf(
            Scope([self.scope.query[i] for i in keep]),
            log_weights=self.log_weights.detach()[keep],
            smoothing=self.smoothing,
        )


def histogram_edges(values: np.ndarray, bins: int, strategy: str) -> np.ndarray:
    """`bins + 1` strictly increasing edges covering [min, max] of `values` (one feature).

    `uniform`: equal widths. `quantile`: equal training mass per bin; edges that coincide (e.g. a point
    mass at SE = 0) are merged and the widest bins split in half until there are `bins` bins again.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    if strategy == "uniform":
        return np.linspace(lo, hi, bins + 1)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    edges[0], edges[-1] = lo, hi
    while len(edges) < bins + 1:
        k = int(np.argmax(np.diff(edges)))
        edges = np.insert(edges, k + 1, (edges[k] + edges[k + 1]) / 2)
    return edges


class HistogramLeaf(LeafModule):
    """Independent piecewise-constant densities, one per variable in scope, with exponential tails.

    Variable v has fixed edges e_0 < … < e_B (from the full training data) and B + 2 cells: the B bins and
    the two tails. With smoothed cell probabilities p (Laplace pseudo-count `smoothing`, so none is 0):

        f(x) = p_k / (e_k − e_{k−1})                          x in bin k (closed at e_B)
        f(x) = p_low  / s · exp(−(e_0 − x) / s)               x < e_0
        f(x) = p_high / s · exp(−(x − e_B) / s)               x > e_B,     s = (e_B − e_0) / B

    Each tail integrates to its p, so f is a proper density on the whole real line: test values outside
    the training range get a small but non-zero density instead of 0.
    """

    def __init__(
        self,
        scope: Scope | int | list[int],
        out_channels: int = 1,
        num_repetitions: int = 1,
        edges: Tensor | None = None,
        log_probs: Tensor | None = None,
        smoothing: float | None = None,
        validate_args: bool | None = None,
    ) -> None:
        if log_probs is not None:
            out_channels, num_repetitions = log_probs.shape[1], log_probs.shape[2]
        super().__init__(scope, out_channels, num_repetitions, validate_args=validate_args)
        if edges is None:
            table = _active().edges
            edges = torch.stack([torch.as_tensor(table[q], dtype=torch.float64) for q in self.scope.query])
        edges = torch.as_tensor(edges, dtype=torch.float64)
        if edges.dim() != 2 or edges.shape[0] != len(self.scope.query) or edges.shape[1] < 2:
            raise ValueError(f"edges must have shape (variables, bins + 1), got {tuple(edges.shape)}")
        if not torch.all(edges[:, 1:] > edges[:, :-1]):
            raise ValueError("histogram edges must be strictly increasing")
        self.smoothing = _active().bin_smoothing if smoothing is None else float(smoothing)
        self.register_buffer("edges", edges)
        cells = edges.shape[1] + 1
        if log_probs is None:
            log_probs = torch.full((*self._event_shape, cells), -math.log(cells))
        self.log_probs = nn.Parameter(torch.as_tensor(log_probs, dtype=torch.float64))

    @property
    def tail_scale(self) -> Tensor:
        return (self.edges[:, -1] - self.edges[:, 0]) / (self.edges.shape[1] - 1)

    def cells(self, x: Tensor) -> Tensor:
        """Cell index per value: 0 = low tail, 1..B = bins, B + 1 = high tail. x: (N, F)."""
        idx = torch.empty_like(x, dtype=torch.long)
        for f in range(x.shape[1]):
            e = self.edges[f].to(x.dtype)
            inner = torch.bucketize(x[:, f].contiguous(), e[1:-1].contiguous(), right=True) + 1
            idx[:, f] = torch.where(x[:, f] < e[0], 0, torch.where(x[:, f] > e[-1], e.numel(), inner))
        return idx

    def log_density(self, x: Tensor) -> Tensor:
        """log f(x) per variable/channel/repetition. x: (N, F) without NaN -> (N, F, C, R)."""
        x = x.to(torch.float64)
        idx = self.cells(x)
        s = self.tail_scale
        log_norm = torch.cat([s[:, None], torch.diff(self.edges, dim=1), s[:, None]], dim=1).log()
        log_norm_x = torch.gather(log_norm.expand(x.shape[0], -1, -1), 2, idx[..., None])[..., 0]
        decay = torch.where(idx == 0, self.edges[:, 0] - x, 0.0) + torch.where(
            idx == log_norm.shape[1] - 1, x - self.edges[:, -1], 0.0
        )
        lp = self.log_probs.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
        lp_x = torch.gather(lp, 4, idx[:, :, None, None, None].expand(-1, -1, lp.shape[2], lp.shape[3], 1))[..., 0]
        return lp_x - (log_norm_x + decay / s)[:, :, None, None]

    def log_likelihood(self, data: Tensor, cache: object = None) -> Tensor:
        x = data[:, self.scope.query].to(torch.float64)
        missing = torch.isnan(x)
        ll = self.log_density(torch.where(missing, self.edges[None, :, 0], x))
        return torch.where(missing[:, :, None, None], torch.zeros_like(ll), ll)

    @property
    def _torch_distribution_class(self) -> None:
        return None

    @property
    def _supported_value(self) -> Tensor:
        return self.edges[:, 0]

    def params(self) -> dict[str, Tensor]:
        return {"log_probs": self.log_probs}

    def _compute_parameter_estimates(self, data: Tensor, weights: Tensor, bias_correction: bool) -> dict[str, Tensor]:
        x = rearrange(data, "n f 1 1 -> n f").to(torch.float64)
        one_hot = nn.functional.one_hot(self.cells(x), self.log_probs.shape[-1]).to(torch.float64)
        counts = torch.einsum("nfk,nfcr->fcrk", one_hot, weights.to(torch.float64))
        a = self.smoothing
        probs = (counts + a) / (counts.sum(-1, keepdim=True) + a * counts.shape[-1])
        return {"log_probs": probs.log()}

    def _set_mle_parameters(self, params_dict: dict[str, Tensor]) -> None:
        self.log_probs.data = params_dict["log_probs"].to(self.log_probs.dtype)

    def marginalize(self, marg_rvs: list[int], prune: bool = True, cache: object = None) -> LeafModule | None:
        keep = [i for i, q in enumerate(self.scope.query) if q not in marg_rvs]
        if not keep:
            return None
        return HistogramLeaf(
            Scope([self.scope.query[i] for i in keep]),
            edges=self.edges[keep],
            log_probs=self.log_probs.detach()[keep],
            smoothing=self.smoothing,
        )


_VARIANCE_FLOOR: ContextVar[float] = ContextVar("spuma_spn_variance_floor", default=0.0)


@contextmanager
def variance_floor(floor: float) -> Iterator[None]:
    """`FlooredNormal` leaves created inside this block keep sigma^2 >= `floor`."""
    token = _VARIANCE_FLOOR.set(float(floor))
    try:
        yield
    finally:
        _VARIANCE_FLOOR.reset(token)


class FlooredNormal(Normal):
    """Gaussian leaf whose MLE variance is floored: sigma^2 = max(sigma_MLE^2, floor). A slice where a feature is
    (nearly) constant would otherwise get a spike density that dominates the likelihood ratio of a class."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.variance_floor = _VARIANCE_FLOOR.get()

    def _compute_parameter_estimates(self, data: Tensor, weights: Tensor, bias_correction: bool) -> dict[str, Tensor]:
        params = super()._compute_parameter_estimates(data, weights, bias_correction)
        params["scale"] = params["scale"].clamp(min=math.sqrt(self.variance_floor))
        return params


CONTINUOUS_LEAVES: dict[str, type[LeafModule]] = {"normal": Normal, "laplace": Laplace, "histogram": HistogramLeaf}
