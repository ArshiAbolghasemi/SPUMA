"""Error-risk models: anything that learns P(Y=1 | Z) from the feature matrix plugs in here.

`evaluation.risk_models` in the config lists which registered models are trained and evaluated. Only the
SPN is registered today; a new model needs a class implementing `RiskModel` and one `RISK_MODELS` entry
(plus its name in `EvaluationConfig.risk_models`' allowed values).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

from spuma.spn.model import ClassConditionalSPN

if TYPE_CHECKING:
    from spuma.config import Settings


class RiskModel(Protocol):
    def fit(self, Z: np.ndarray, y: np.ndarray) -> Self: ...

    def predict_error_proba(self, Z: np.ndarray) -> np.ndarray:
        """P(Y=1 | Z) for each row (Y=1: the final answer is wrong)."""
        ...

    def save(self, path: Path) -> None: ...

    def summary(self) -> dict[str, Any]:
        """JSON-able training summary for the run's status file."""
        ...


@dataclass(frozen=True)
class RiskModelSpec:
    build: Callable[[Settings], RiskModel]
    load: Callable[[Path], RiskModel]
    suffix: str = ".pt"


RISK_MODELS: dict[str, RiskModelSpec] = {
    "spn": RiskModelSpec(
        build=lambda s: ClassConditionalSPN(s.spn, s.features.order),
        load=ClassConditionalSPN.load,
    ),
}


def risk_model_spec(name: str) -> RiskModelSpec:
    if name not in RISK_MODELS:
        raise ValueError(f"unknown risk model {name!r}; registered: {sorted(RISK_MODELS)}")
    return RISK_MODELS[name]
