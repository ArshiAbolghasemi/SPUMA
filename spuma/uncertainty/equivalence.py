"""Answer-aware equivalence for semantic entropy: math-verify for boxed answers, NLI for everything else.

UQLM's clusterer asks its NLI object, for pairs of texts, whether they mean the same thing
(`get_nli_results_batch`). `AnswerAwareNLI` answers through that same interface:
  - both texts contain a \\boxed{} answer: equivalent iff the last answers are equivalent -- exact, and immune
    to NLI truncation. The answer comparison depends on the dataset (`answer_equivalence`):
      MATH (default): math-verify in either direction (like UQLM's either-direction entailment);
      MoreHopQA: answers with a word of 3+ letters (outside LaTeX commands) are text, equal after SQuAD
      normalization or as the same date under dateutil; other answers go to math-verify. Math-verify treats
      letters as commuting symbols ("listen" == "silent"), so short-answer QA keeps text away from it; MATH
      does not, because its text-looking answers (units, `pmatrix`, "0 and 4") are math;
      MMLU, MuSiQue, MedHopQA: the answer is the last `ANSWER:` / `VERIFIED_ANSWER:` / `SUGGESTED_ANSWER:` line
      instead of \\boxed{} (`answer_extractor`); MMLU answers are equal as the same option letter, MuSiQue and
      MedHopQA answers after SQuAD normalization;
  - exactly one does: not equivalent (one commits to an answer, the other does not);
  - neither does (unfinished reasoning, no boxed answer): the wrapped NLI model decides, unchanged.
Samples are only compared with each other; the reference answer is never involved.
"""

import logging
import re
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from typing import Any

from dateutil import parser as date_parser
from math_verify import parse, verify

from spuma.dataset.base import normalize
from spuma.dataset.math.answers import extract_boxed
from spuma.dataset.morehopqa.grading import clean, normalize_answer
from spuma.dataset.qa import any_answer, letter_equivalent, text_equivalent

logger = logging.getLogger(__name__)

MATH_VERIFY_TIMEOUT_S = 5
LATEX_COMMAND_RE = re.compile(r"\\[A-Za-z]+")
WORD_RE = re.compile(r"[A-Za-z]{3,}")
DATE_DEFAULTS = (datetime(1, 1, 1), datetime(2, 2, 2, 2, 2))


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
    if not pa or not pb:
        return False
    return bool(
        verify(pa, pb, timeout_seconds=MATH_VERIFY_TIMEOUT_S) or verify(pb, pa, timeout_seconds=MATH_VERIFY_TIMEOUT_S)
    )


def is_text_answer(answer: str) -> bool:
    return bool(WORD_RE.search(LATEX_COMMAND_RE.sub(" ", clean(answer))))


def _same_date(a: str, b: str) -> bool:
    try:
        return all(date_parser.parse(a, default=d) == date_parser.parse(b, default=d) for d in DATE_DEFAULTS)
    except (ValueError, OverflowError):
        return False


@lru_cache(maxsize=65536)
def qa_answers_equivalent(a: str, b: str) -> bool:
    if not (is_text_answer(a) or is_text_answer(b)):
        return answers_equivalent(a, b)
    a, b = clean(a), clean(b)
    return normalize_answer(a) == normalize_answer(b) or _same_date(a, b)


ANSWER_EQUIVALENCE: dict[str, Callable[[str, str], bool]] = {
    "morehopqa": qa_answers_equivalent,
    "mmlu": letter_equivalent,
    "musique": text_equivalent,
    "medhopqa": text_equivalent,
}
ANSWER_EXTRACTORS: dict[str, Callable[[str], str | None]] = {
    "mmlu": any_answer,
    "musique": any_answer,
    "medhopqa": any_answer,
}


def answer_equivalence(dataset: str) -> Callable[[str, str], bool]:
    return ANSWER_EQUIVALENCE.get(dataset, answers_equivalent)


def answer_extractor(dataset: str) -> Callable[[str], str | None]:
    return ANSWER_EXTRACTORS.get(dataset, boxed_answer)


def _result(same: bool) -> dict[str, Any]:
    return {"noncontradiction_score": float(same), "entailment": same, "entailment_score": float(same)}


class AnswerAwareNLI:
    """UQLM NLI interface: `equivalent` (default math-verify) for pairs of answered texts, the wrapped NLI for
    the rest.

    `nli_factory` builds the NLI model on first need, so runs whose texts all box an answer never load it.
    """

    def __init__(
        self,
        nli_factory: Callable[[], Any],
        equivalent: Callable[[str, str], bool] = answers_equivalent,
        extract: Callable[[str], str | None] = boxed_answer,
    ) -> None:
        self._factory = nli_factory
        self._equivalent = equivalent
        self._extract = extract
        self._nli: Any = None
        self.num_responses: int | None = None
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
            ans_a, ans_b = self._extract(a), self._extract(b)
            if ans_a is not None and ans_b is not None:
                out[i] = _result(self._equivalent(ans_a, ans_b))
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
