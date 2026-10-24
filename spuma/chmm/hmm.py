"""The CHMM: hmmlearn's `GaussianHMM` over per-response situational distances x_t = [d_q(t), d_i(t)].

    S_1 ~ pi,  S_t | S_{t-1} ~ A,  z_t | S_t = k ~ N(mu_k, diag(var_k)),  z_t = standardize(x_t)

The scaler and the HMM are fitted on the training trajectories only, one sequence per trajectory (`lengths`
keeps the boundaries). hmmlearn cannot integrate a missing value out, so a missing interaction gap (the first
response) is filled with the *training mean* gap, which standardizes to 0: an average gap, neither agreement nor
disagreement. After fitting, states are ordered by deviation, the mean of mu_k over both features in
training-standard-deviation units, so state 0/1/2 is always low/moderate/high deviation; gamma_t(k) =
P(S_t = k | x_1:T) comes from `predict_proba` with the columns in that order.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Self

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from spuma.chmm.config import HMMConfig

logger = logging.getLogger(__name__)

STATE_NAMES = ("low", "moderate", "high")


class SituationalHMM:
    def __init__(self, config: HMMConfig) -> None:
        self.config = config
        self.fill: np.ndarray | None = None
        self.scaler: StandardScaler | None = None
        self.model: GaussianHMM | None = None
        self.order: np.ndarray | None = None

    def _scaled(self, seqs: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
        X = np.concatenate(seqs)
        X = np.where(np.isnan(X), self.fill, X)
        return self.scaler.transform(X), [len(s) for s in seqs]

    def fit(self, seqs: list[np.ndarray]) -> Self:
        c = self.config
        X = np.concatenate(seqs)
        self.fill = np.nanmean(X, 0)
        self.scaler = StandardScaler().fit(np.where(np.isnan(X), self.fill, X))
        Z, lengths = self._scaled(seqs)
        self.model = GaussianHMM(
            n_components=c.n_components, covariance_type=c.covariance_type, n_iter=c.n_iter, random_state=c.random_state
        ).fit(Z, lengths)
        # a state never left in training keeps an all-zero transmat_ row, which hmmlearn then refuses to score
        empty = self.model.transmat_.sum(1) == 0
        if empty.any():
            logger.warning(
                "CHMM states without observed transitions get uniform rows", extra={"states": empty.tolist()}
            )
            self.model.transmat_[empty] = 1 / c.n_components
        self.order = np.argsort(self.model.means_.mean(1), kind="stable")
        return self

    @property
    def deviation(self) -> np.ndarray:
        return self.model.means_.mean(1)[self.order]

    def posteriors(self, seqs: list[np.ndarray]) -> list[np.ndarray]:
        """gamma (T, 3) per sequence, states low -> high deviation."""
        if self.model is None:
            raise RuntimeError("CHMM is not fitted")
        Z, lengths = self._scaled(seqs)
        gamma = self.model.predict_proba(Z, lengths)[:, self.order]
        return np.split(gamma, np.cumsum(lengths)[:-1])

    def describe(self) -> dict[str, Any]:
        """Parameters in state order; means and variances in the original distance units."""
        m, o, s = self.model, self.order, self.scaler.scale_
        return {
            "states": list(STATE_NAMES),
            "startprob": m.startprob_[o].tolist(),
            "transmat": m.transmat_[np.ix_(o, o)].tolist(),
            "means": self.scaler.inverse_transform(m.means_[o]).tolist(),
            "variances": (np.stack([np.diag(c) for c in m.covars_])[o] * s**2).tolist(),
            "deviation": self.deviation.tolist(),
            "missing_fill": self.fill.tolist(),
            "converged": bool(m.monitor_.converged),
            "iterations": int(m.monitor_.iter),
            "log_likelihood": float(m.monitor_.history[-1]),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {"config": self.config, "fill": self.fill, "scaler": self.scaler, "model": self.model}
        path.write_bytes(pickle.dumps({**state, "order": self.order}))

    @classmethod
    def load(cls, path: Path) -> Self:
        state = pickle.loads(path.read_bytes())
        hmm = cls(state["config"])
        hmm.fill, hmm.scaler, hmm.model, hmm.order = state["fill"], state["scaler"], state["model"], state["order"]
        return hmm


def state_coefficients(alpha: float, states: int = 3) -> np.ndarray:
    """lambda(alpha) = [1, 1 + alpha, 1 + 2 alpha]: fixed, monotone in deviation, never learned from labels."""
    return 1.0 + alpha * np.arange(states)


def step_weights(gamma: np.ndarray, alpha: float) -> np.ndarray:
    """w_t = r_t / sum_j r_j,  r_t = sum_k lambda_k(alpha) gamma_t(k)."""
    r = gamma @ state_coefficients(alpha, gamma.shape[1])
    return r / r.sum()
