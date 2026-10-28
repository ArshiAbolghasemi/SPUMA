"""MoreHopQA grading of the final answer against the reference (evaluation side only, never exposed to agents).

The final answer is the last \\boxed{} of the final message (the MATH-style prompts ask for \\boxed{}), else its
last `ANSWER:` line (the question-answering prompts' format); without either the answer is graded incorrect and
flagged `parse_failure`. A boxed answer is correct when, after removing
LaTeX text wrappers, it matches the reference:
  - number: equal numeric value (thousands separators ignored), else Math-Verify equivalence;
  - date / datetime: the same calendar date (and minute, for datetime) after parsing both;
  - otherwise: SQuAD-style normalized exact match (case, punctuation, articles, whitespace).
Label Y = 1 if the answer is incorrect.
"""

import logging
import re
import string
from datetime import datetime

from dateutil import parser as date_parser
from math_verify import parse, verify

from spuma.dataset.base import Grade, Problem
from spuma.dataset.math.answers import extract_boxed

logger = logging.getLogger(__name__)

TEXT_WRAPPER_RE = re.compile(r"\\(?:text|textbf|mathrm|mathbf|operatorname)\{([^{}]*)\}")
ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def clean(answer: str) -> str:
    """Boxed content without LaTeX text wrappers, math delimiters or surrounding quotes."""
    previous = None
    while previous != answer:
        previous, answer = answer, TEXT_WRAPPER_RE.sub(r"\1", answer)
    return answer.replace("$", "").strip().strip("\"'").strip()


def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(ARTICLES_RE.sub(" ", text).split())


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").strip())
    except ValueError:
        return None


def _numbers_match(predicted: str, reference: str, timeout: int) -> bool:
    a, b = _number(predicted), _number(reference)
    if a is not None and b is not None:
        return a == b
    pa = parse(f"\\boxed{{{predicted}}}", parsing_timeout=timeout)
    pb = parse(f"\\boxed{{{reference}}}", parsing_timeout=timeout)
    return bool(pa) and bool(pb) and bool(verify(pb, pa, timeout_seconds=timeout))


def _datetimes_match(predicted: str, reference: str, with_time: bool) -> bool:
    try:
        ref = datetime.fromisoformat(reference.strip())
        pred = date_parser.parse(predicted, default=ref.replace(hour=0, minute=0, second=0, microsecond=0))
    except (ValueError, OverflowError):
        return False
    if with_time:
        return (pred.date(), pred.hour, pred.minute) == (ref.date(), ref.hour, ref.minute)
    return pred.date() == ref.date()


def answer_matches(predicted: str, reference: str, answer_type: str | None, timeout: int = 5) -> tuple[bool, str]:
    """(correct, method) for a cleaned predicted answer against the reference."""
    if normalize_answer(predicted) == normalize_answer(reference):
        return True, "exact"
    if answer_type == "number":
        return _numbers_match(predicted, reference, timeout), "number"
    if answer_type in ("date", "datetime"):
        return _datetimes_match(predicted, reference, answer_type == "datetime"), answer_type
    return False, "exact"


class MoreHopQAGrader:
    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout = timeout_seconds

    def grade(self, problem: Problem, final_text: str) -> Grade:
        boxed = extract_boxed(final_text)
        if boxed is None:
            from spuma.dataset.qa import final_answer  # noqa: PLC0415 - qa imports this module

            boxed = final_answer(final_text)
        predicted = clean(boxed) if boxed is not None else None
        reference = problem.reference_solution
        if predicted is None:
            correct, method = False, "no_answer"
        else:
            correct, method = answer_matches(predicted, reference, problem.metadata.get("answer_type"), self.timeout)
        grade = Grade(
            problem_id=problem.problem_id,
            predicted=predicted,
            reference=reference,
            correct=correct,
            label=0 if correct else 1,
            parse_failure=predicted is None,
            reference_parse_failure=False,
            method=method,
        )
        logger.info(
            "graded",
            extra={"problem_id": problem.problem_id, "label": grade.label, "parse_failure": grade.parse_failure},
        )
        return grade
