import dataclasses

import pytest

from spuma.dataset.base import Problem
from spuma.dataset.morehopqa import MoreHopQADataset, MoreHopQAGrader, answer_matches

ROWS = [
    {"question": f"Q{i}?", "answer": str(i), "answer_type": "number", "no_of_hops": 2, "reasoning_type": "Arithmetic"}
    for i in range(30)
]


@pytest.fixture
def morehopqa(config, set_settings):
    def install(**fields):
        defaults = {
            "name": "morehopqa",
            "hf_path": "alabnii/morehopqa",
            "revision": "abc",
            "data_files": "verified/test.parquet",
            "train_samples": 0,
            "validation_samples": 0,
            "test_samples": 10,
        }
        dataset = dataclasses.replace(config.dataset, **(defaults | fields))
        return set_settings(dataclasses.replace(config, dataset=dataset))

    return install


def test_loads_a_seeded_test_only_sample_from_the_pinned_parquet(morehopqa):
    morehopqa()
    urls = []
    splits = MoreHopQADataset(loader=lambda url: urls.append(url) or ROWS).load()
    assert urls == ["hf://datasets/alabnii/morehopqa@abc/verified/test.parquet"]
    assert splits["train"] == splits["validation"] == [] and len(splits["test"]) == 10
    first = splits["test"][0]
    assert first.problem_id.startswith("morehopqa/test/") and first.split == "test"
    assert first.problem == ROWS[int(first.problem_id.rsplit("/", 1)[1])]["question"]
    assert first.metadata == {"answer_type": "number", "no_of_hops": 2, "reasoning_type": "Arithmetic"}
    again = MoreHopQADataset(loader=lambda url: ROWS).load()["test"]
    assert [p.problem_id for p in again] == [p.problem_id for p in splits["test"]]


def test_refuses_train_splits_and_oversized_samples(morehopqa):
    morehopqa(train_samples=5)
    with pytest.raises(ValueError, match="evaluation-only"):
        MoreHopQADataset(loader=lambda url: ROWS).load()
    morehopqa(test_samples=31)
    with pytest.raises(ValueError, match="requested 31"):
        MoreHopQADataset(loader=lambda url: ROWS).load()


@pytest.mark.parametrize(
    ("predicted", "reference", "answer_type", "correct"),
    [
        ("4", "4", "number", True),
        ("4.0", "4", "number", True),
        ("2,011", "2011", "number", True),
        ("5", "4", "number", False),
        ("September 11, 1986", "1986-09-11", "date", True),
        ("11 Sep 1986", "1986-09-11", "date", True),
        ("1986-09-12", "1986-09-11", "date", False),
        ("June 4, 1973 11:43", "1973-06-04 11:43", "datetime", True),
        ("1973-06-04 11:44", "1973-06-04 11:43", "datetime", False),
        ("The Tuesday", "Tuesday", "string", True),
        ("N", "n", "letter", True),
        ("o", "n", "letter", False),
    ],
)
def test_answer_matching_by_type(predicted, reference, answer_type, correct):
    assert answer_matches(predicted, reference, answer_type)[0] is correct


def test_grader_reads_the_last_boxed_answer_and_flags_missing_ones():
    problem = Problem("morehopqa/test/1", "Who?", "Martha Coolidge", "test", {"answer_type": "string"})
    grade = MoreHopQAGrader().grade(problem, "First \\boxed{x}. So the director is \\boxed{\\text{Martha Coolidge}}.")
    assert grade.correct and grade.label == 0 and grade.predicted == "Martha Coolidge" and grade.method == "exact"
    missing = MoreHopQAGrader().grade(problem, "Martha Coolidge")
    assert missing.label == 1 and missing.parse_failure and missing.method == "no_answer"
