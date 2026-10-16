"""Answer-aware equivalence for semantic entropy: math-verify for boxed answers, NLI for everything else.

UQLM's clusterer asks its NLI object, for pairs of texts, whether they mean the same thing
(`get_nli_results_batch`). `AnswerAwareNLI` answers through that same interface:
  - both texts contain a \\boxed{} answer: equivalent iff the last answers are equivalent by math-verify
    (either direction, like UQLM's either-direction entailment) -- exact, and immune to NLI truncation;
  - exactly one does: not equivalent (one commits to an answer, the other does not);
  - neither does (unfinished reasoning, no boxed answer): the wrapped NLI model decides, unchanged.
Samples are only compared with each other; the reference answer is never involved.
"""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from math_verify import parse, verify

from spuma.dataset.base import normalize
from spuma.dataset.math.answers import extract_boxed

logger = logging.getLogger(__name__)

MATH_VERIFY_TIMEOUT_S = 5


@lru_cache(maxsize=65536)
def boxed_answer(text: str) -> str | None:
    return extract_boxed(text)


@lru_cache(maxsize=65536)
def _parsed(answer: str) -> Any:
    return parse(f"\\boxed{{{answer}}}", parsing_timeout=MATH_VERIFY_TIMEOUT_S)


@lru_cache(maxsize=65536)
def answers_equivalent(a: str, b: str) -> bool:
    if normalize(a) == normalize(b):
        return True
    pa, pb = _parsed(a), _parsed(b)
    if not pa or not pb:  # unparseable: only identical (normalized) answers match
        return False
    return bool(
        verify(pa, pb, timeout_seconds=MATH_VERIFY_TIMEOUT_S) or verify(pb, pa, timeout_seconds=MATH_VERIFY_TIMEOUT_S)
    )


def _result(same: bool) -> dict[str, Any]:
    return {"noncontradiction_score": float(same), "entailment": same, "entailment_score": float(same)}


class AnswerAwareNLI:
    """UQLM NLI interface: math-verify for pairs of answered texts, the wrapped NLI for the rest.

    `nli_factory` builds the NLI model on first need, so runs whose texts all box an answer never load it.
    """

    def __init__(self, nli_factory: Callable[[], Any]) -> None:
        self._factory = nli_factory
        self._nli: Any = None
        self.num_responses: int | None = None  # set by UQLM before each scoring call
        self.counts = {"math_verify": 0, "answer_vs_none": 0, "nli": 0}

    @property
    def nli(self) -> Any:
        if self._nli is None:
            self._nli = self._factory()
        self._nli.num_responses = self.num_responses
        return self._nli

    def get_nli_results_batch(self, response_pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any] | None] = [None] * len(response_pairs)
        pending: list[int] = []
        for i, (a, b) in enumerate(response_pairs):
            ans_a, ans_b = boxed_answer(a), boxed_answer(b)
            if ans_a is not None and ans_b is not None:
                out[i] = _result(answers_equivalent(ans_a, ans_b))
                self.counts["math_verify"] += 1
            elif (ans_a is None) != (ans_b is None):
                out[i] = _result(False)
                self.counts["answer_vs_none"] += 1
            else:
                pending.append(i)
        if pending:
            self.counts["nli"] += len(pending)
            for i, res in zip(
                pending, self.nli.get_nli_results_batch([response_pairs[i] for i in pending]), strict=True
            ):
                out[i] = res
        return out  # type: ignore[return-value]
