"""MoreHopQA (Schnitzler et al., 2024): multi-hop questions with a reasoning step on the last hop; loader and grader."""

from spuma.dataset.morehopqa.grading import MoreHopQAGrader, answer_matches
from spuma.dataset.morehopqa.loader import MoreHopQADataset

__all__ = ["MoreHopQADataset", "MoreHopQAGrader", "answer_matches"]
