"""Offline NLI stand-in for every test (registered in pyproject: `-p spuma.uncertainty.tests.fixtures`)."""

import re
import time
import types

import pytest
import uqlm.scorers.shortform.entropy as entropy

from spuma.uncertainty import estimators


class RuleNLI:
    """Stand-in for uqlm.nli.NLI with the same interface: "entailment" iff both texts end in the same number.

    UQLM's BlackBoxUQ, clustering and entropy code run unchanged on top of it; only the DeBERTa download
    is avoided.
    """

    def __init__(self):
        self.num_responses = None
        self.probabilities = {}
        self.pairs: list[tuple[str, str]] = []

    @staticmethod
    def _final_number(text: str):

        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[-1]) if nums else None

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

    # UQLM's SemanticEntropy.score ends with a fixed time.sleep(0.1); skip it in tests only
    monkeypatch.setattr(entropy, "time", types.SimpleNamespace(sleep=lambda s: None, time=time.time))
    # real UQLM scorer, clustering and entropy; rule NLI instead of the model download
    monkeypatch.setattr(estimators, "LabelOrderedNLI", lambda **kwargs: nli)
    return nli
