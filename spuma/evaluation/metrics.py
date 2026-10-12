"""Evaluation metrics registry. Every metric takes (y, p): y = 1 if the answer is incorrect, p = predicted
error risk P(Y=1|Z) (any score where higher = more likely wrong works for the ranking metrics).

Risk-coverage convention (AURC / AUARC):
  - samples are rejected in order of *highest* predicted error risk first, i.e. kept in ascending p;
  - coverage takes the n values k/n, k = 1..n; risk(k) = error rate among the k kept samples;
  - the area is the left-Riemann mean over that grid: AURC = (1/n) Σ_k risk(k), AUARC = (1/n) Σ_k (1 - risk(k)),
    so AUARC = 1 - AURC; an oracle ranking gives the maximum AUARC for the given error rate;
  - ties in p use the expected risk under a uniformly random order within the tie group.
"""

import logging
from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)

EvaluationMetric = Callable[[np.ndarray, np.ndarray], float]
METRICS: dict[str, EvaluationMetric] = {}


def register_metric(name: str) -> Callable[[EvaluationMetric], EvaluationMetric]:
    def deco(fn: EvaluationMetric) -> EvaluationMetric:
        METRICS[name] = fn
        return fn

    return deco


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """H(Y|Z) = -p log p - (1-p) log(1-p) of a binary posterior p = P(Y=1|Z), in nats."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


def _both_classes(y: np.ndarray, name: str) -> bool:
    if len(np.unique(y)) < 2:
        logger.debug("metric undefined with a single class", extra={"metric": name})
        return False
    return True


@register_metric("auroc")
def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if _both_classes(y, "auroc") else float("nan")


@register_metric("auprc")
def auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if _both_classes(y, "auprc") else float("nan")


def risk_coverage(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coverage k/n, risk(k)) for k = 1..n, keeping the lowest-risk samples first (tie-aware)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n = len(y)
    order = np.argsort(p, kind="stable")
    ys, ps = y[order], p[order]
    expected_errors = np.empty(n)
    start = 0
    done = 0.0
    while start < n:
        end = start + 1
        while end < n and ps[end] == ps[start]:
            end += 1
        rate = ys[start:end].mean()
        expected_errors[start:end] = done + rate * np.arange(1, end - start + 1)
        done += ys[start:end].sum()
        start = end
    k = np.arange(1, n + 1)
    return k / n, expected_errors / k


@register_metric("aurc")
def aurc(y: np.ndarray, p: np.ndarray) -> float:
    return float(risk_coverage(y, p)[1].mean())


@register_metric("auarc")
def auarc(y: np.ndarray, p: np.ndarray) -> float:
    return 1.0 - aurc(y, p)


@register_metric("accuracy")
def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Final-answer accuracy of the multi-agent system (independent of p)."""
    return float(1.0 - np.mean(y))


@register_metric("brier")
def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


@register_metric("nll")
def nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Σ_b |B_b|/n · |mean(y in B_b) - mean(p in B_b)| over equal-width bins of P(error)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx)))


METRICS["ece"] = expected_calibration_error


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: EvaluationMetric,
    groups: np.ndarray,
    iterations: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI from resampling *groups* (problem ids) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    stats = []
    for _ in range(iterations):
        rows = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        v = metric(y[rows], p[rows])
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    a = (1 - confidence) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    names: list[str],
    groups: np.ndarray | None = None,
    bootstrap: dict | None = None,
    ece_bins: int = 15,
) -> dict[str, dict[str, float]]:
    """{metric: {"value": v, ["ci_low", "ci_high"]}}; `bootstrap` = {"iterations", "confidence", "seed"}."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    out = {}
    for name in names:
        if name not in METRICS:
            raise ValueError(f"unknown metric {name!r}; registered: {sorted(METRICS)}")
        fn = (lambda a, b: expected_calibration_error(a, b, ece_bins)) if name == "ece" else METRICS[name]
        res = {"value": fn(y, p)}
        if bootstrap:
            res["ci_low"], res["ci_high"] = bootstrap_ci(y, p, fn, groups, **bootstrap)
        out[name] = res
    return out
