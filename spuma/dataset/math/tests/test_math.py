import dataclasses

import pytest

from spuma.dataset.base import Problem
from spuma.dataset.math import MathDataset, MathVerifyGrader, extract_boxed


def _fake_loader(path, subset, split):
    n = 6 if split == "train" else 4
    return [
        {"problem": f"{subset} {split} {i}", "solution": f"so \\boxed{{{i}}}", "level": "Level 1", "type": subset}
        for i in range(n)
    ]


def test_loader_ids_metadata_and_splits(config, set_settings):
    sizes = {"subsets": ["algebra", "geometry"], "train_samples": 5, "validation_samples": 3, "test_samples": 4}
    set_settings(dataclasses.replace(config, dataset=dataclasses.replace(config.dataset, **sizes)))
    splits = MathDataset(loader=_fake_loader).load()
    assert {k: len(v) for k, v in splits.items()} == {"train": 5, "validation": 3, "test": 4}
    p = splits["test"][0]
    assert p.problem_id.startswith("math/test/") and p.metadata["level"] == "Level 1"
    assert p.metadata["subject"] in ("algebra", "geometry") and "boxed" in p.reference_solution


def test_extract_boxed():
    assert extract_boxed("a \\boxed{\\frac{1}{2}} b \\boxed{x^{2}}") == "x^{2}"
    assert extract_boxed("none") is None and extract_boxed("\\boxed{unbalanced") is None


def _problem(solution):
    return Problem("p", "q", solution, "test")


@pytest.mark.parametrize(
    ("final", "solution", "label"),
    [
        ("The answer is \\boxed{\\frac{1}{2}}", "\\boxed{0.5}", 0),
        ("\\boxed{x+1}", "thus \\boxed{1+x}", 0),
        ("\\boxed{3}", "\\boxed{4}", 1),
        ("\\boxed{\\dfrac{\\sqrt{2}}{2}}", "\\boxed{\\frac{1}{\\sqrt{2}}}", 0),
    ],
)
def test_grading_labels(final, solution, label):
    g = MathVerifyGrader().grade(_problem(solution), final)
    assert g.label == label and g.correct == (label == 0) and not g.parse_failure


def test_parse_failure_tracked():
    g = MathVerifyGrader().grade(_problem("\\boxed{2}"), "I think it is two.")
    assert g.label == 1 and g.parse_failure and g.method == "no_answer"
