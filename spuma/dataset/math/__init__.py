"""MATH benchmark: loader, \\boxed{} answer extraction, Math-Verify grading."""

from spuma.dataset.math.answers import extract_boxed
from spuma.dataset.math.grading import MathVerifyGrader
from spuma.dataset.math.loader import MathDataset

__all__ = ["MathDataset", "MathVerifyGrader", "extract_boxed"]
