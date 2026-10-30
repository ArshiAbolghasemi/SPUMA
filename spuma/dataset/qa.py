"""Answer extraction and grading shared by the question-answering datasets (MMLU, MuSiQue; MoreHopQA's grader
reads `ANSWER:` lines through `final_answer` when there is no \\boxed{} answer).

Their prompts end every answer-giving response with an `ANSWER: <answer>` line (verifiers and reviewers write
`VERIFIED_ANSWER:` / `SUGGESTED_ANSWER:`). The graded answer is the last `ANSWER:` line of the final response;
`any_answer` also accepts the verifier/reviewer forms (semantic-entropy clustering compares the answers of
samples of the same response). A problem without a reference answer (a blind-test question) is never graded right
or wrong: its grade has `label = None`, and no synthetic label is ever made.
"""

import logging
import re
from collections.abc import Callable

from spuma.dataset.base import Grade, Problem, normalize_answer

logger = logging.getLogger(__name__)

INSUFFICIENT = "INSUFFICIENT_INFORMATION"
_LINE = r"(?im)^[\s>*#_\-]*{key}[*_]*\s*:\s*[*_]*\s*(.*\S)"
ANSWER_RE = re.compile(_LINE.format(key="ANSWER"))
ANY_ANSWER_RE = re.compile(_LINE.format(key="(?:VERIFIED_|SUGGESTED_)?ANSWER"))
LETTER_RE = re.compile(r"^(?:option\s+|choice\s+)?\(?([ABCD])\)?(?=$|[\s.:),])", re.IGNORECASE)


def _clean(answer: str) -> str:
    answer = answer.replace("TASK_DONE", "").strip()
    return answer.strip("*_`").strip().strip("\"'").rstrip(".").strip()


def final_answer(text: str) -> str | None:
    """The last `ANSWER:` line's answer, or None."""
    found = ANSWER_RE.findall(text)
    return (_clean(found[-1]) or None) if found else None


def any_answer(text: str) -> str | None:
    """The last `ANSWER:` / `VERIFIED_ANSWER:` / `SUGGESTED_ANSWER:` line's answer, or None."""
    found = ANY_ANSWER_RE.findall(text)
    return (_clean(found[-1]) or None) if found else None


def option_letter(answer: str) -> str | None:
    """A, B, C or D from an answer such as "B", "(B)", "B. Paris" or "Option B"; None otherwise."""
    m = LETTER_RE.match(answer.strip())
    return m.group(1).upper() if m else None


def is_insufficient(answer: str) -> bool:
    """INSUFFICIENT_INFORMATION in any spelling ("insufficient information", "Insufficient-Information")."""
    return re.sub(r"[\W_]+", "", answer.lower()) == "insufficientinformation"


def text_equivalent(a: str, b: str) -> bool:
    """SQuAD-normalized equality (case, punctuation, articles, whitespace)."""
    return normalize_answer(a) == normalize_answer(b)


def letter_equivalent(a: str, b: str) -> bool:
    la, lb = option_letter(a), option_letter(b)
    return la is not None and la == lb


class QAGrader:
    """Grades the final response's `ANSWER:` with `match(predicted, problem) -> (correct, method)`."""

    def __init__(self, match: Callable[[str, Problem], tuple[bool, str]]) -> None:
        self.match = match

    def grade(self, problem: Problem, final_text: str) -> Grade:
        predicted = final_answer(final_text)
        if problem.reference_solution is None:
            correct, method = None, "unlabeled"
        elif predicted is None:
            correct, method = False, "no_answer"
        else:
            correct, method = self.match(predicted, problem)
        grade = Grade(
            problem_id=problem.problem_id,
            predicted=predicted,
            reference=problem.reference_solution,
            correct=correct,
            label=None if correct is None else int(not correct),
            parse_failure=predicted is None,
            reference_parse_failure=False,
            method=method,
        )
        logger.info(
            "graded",
            extra={"problem_id": problem.problem_id, "label": grade.label, "parse_failure": grade.parse_failure},
        )
        return grade
