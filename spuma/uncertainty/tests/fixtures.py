"""Offline NLI stand-in for every test (registered in pyproject: `-p spuma.uncertainty.tests.fixtures`)."""

import re
import time
import types

import numpy as np
import pytest
import uqlm.scorers.shortform.density as density
import uqlm.scorers.shortform.entropy as entropy

from spuma.uncertainty import estimators


class RuleNLI:
    """Stand-in for uqlm.nli.NLI with the same interface: "entailment" iff both texts end in the same number.

    UQLM's white-box scorers (semantic entropy clustering, semantic density) run unchanged on top of it; only
    the NLI model download is avoided. `predict_batch` gives the same rule as probabilities (entailment 1 or
    contradiction 1).
    """

    label_mapping = ["contradiction", "neutral", "entailment"]

    def __init__(self):
        self.num_responses = None
        self.probabilities = {}
        self.pairs: list[tuple[str, str]] = []

    @staticmethod
    def _final_number(text: str):

        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[-1]) if nums else None

    def _same(self, a: str, b: str) -> bool:
        return a == b or (self._final_number(a) is not None and self._final_number(a) == self._final_number(b))

    def predict_batch(self, pairs):
        self.pairs += list(pairs)
        return np.array([[0.0, 0.0, 1.0] if self._same(a, b) else [1.0, 0.0, 0.0] for a, b in pairs])

    def get_nli_results_batch(self, response_pairs):
        self.pairs += list(response_pairs)
        out = []
        for a, b in response_pairs:
            same = a == b or (self._final_number(a) is not None and self._final_number(a) == self._final_number(b))
            out.append(
                {"noncontradiction_score": 1.0 if same else 0.0, "entailment": same, "entailment_score": float(same)}
            )
        return out


@pytest.fixture(autouse=True)
def rule_nli(request, monkeypatch):
    """Offline tests never load the real NLI model; `nli`-marked tests do."""
    if request.node.get_closest_marker("nli"):
        return None
    nli = RuleNLI()

    monkeypatch.setattr(entropy, "time", types.SimpleNamespace(sleep=lambda s: None, time=time.time))
    monkeypatch.setattr(density, "time", types.SimpleNamespace(sleep=lambda s: None, time=time.time))
    monkeypatch.setattr(estimators, "LabelOrderedNLI", lambda **kwargs: nli)
    return nli
