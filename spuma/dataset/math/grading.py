"""MATH grading against the reference \\boxed{} answer (evaluation side only, never exposed to agents).

Label Y = 1 if the final (Reviewer) answer is incorrect. A final message without a \\boxed{} answer is
graded incorrect and flagged `parse_failure` so it can be analysed separately.
"""

import logging

from math_verify import parse, verify

from spuma.dataset.base import Grade, Problem, normalize
from spuma.dataset.math.answers import extract_boxed

logger = logging.getLogger(__name__)


class MathVerifyGrader:
    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout = timeout_seconds

    def grade(self, problem: Problem, final_text: str) -> Grade:
        predicted, reference = extract_boxed(final_text), extract_boxed(problem.reference_solution)
        ref_parsed = parse(f"\\boxed{{{reference}}}", parsing_timeout=self.timeout) if reference else []
        if predicted is None:
            correct, method = False, "no_answer"
        elif ref_parsed:
            pred_parsed = parse(f"\\boxed{{{predicted}}}", parsing_timeout=self.timeout)
            correct = bool(pred_parsed) and bool(verify(ref_parsed, pred_parsed, timeout_seconds=self.timeout))
            method = "math_verify"
        else:
            correct, method = reference is not None and normalize(predicted) == normalize(reference), "string"
        grade = Grade(
            problem_id=problem.problem_id,
            predicted=predicted,
            reference=reference,
            correct=correct,
            label=0 if correct else 1,
            parse_failure=predicted is None,
            reference_parse_failure=not ref_parsed,
            method=method,
        )
        logger.info(
            "graded",
            extra={"problem_id": problem.problem_id, "label": grade.label, "parse_failure": grade.parse_failure},
        )
        return grade
