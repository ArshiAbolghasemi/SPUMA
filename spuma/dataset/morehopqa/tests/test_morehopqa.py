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
    assert first.metadata["answer_type"] == "number" and first.metadata["grading"]["reasoning_type"] == "Arithmetic"
    assert first.metadata["grading"]["human_verified"]
    again = MoreHopQADataset(loader=lambda url: ROWS).load()["test"]
    assert [p.problem_id for p in again] == [p.problem_id for p in splits["test"]]


TRAIN = [
    {
        "_id": f"t{i}",
        "question": f"T{i}?",
        "answer": str(i),
        "answer_type": "number",
        "previous_question": f"First hop {i}?",
    }
    for i in range(12)
]


def test_trains_on_unverified_questions_that_share_no_first_hop_with_the_verified_set(morehopqa):
    verified = [
        {**r, "previous_question": "First hop 0?" if r["question"] == "Q0?" else f"Other {r['question']}"} for r in ROWS
    ]
    morehopqa(
        train_samples=6, validation_samples=4, data_files={"test": "verified/test.parquet", "train": "data/u.json"}
    )
    urls = []
    loader = lambda url: urls.append(url) or (TRAIN if url.endswith(".json") else verified)  # noqa: E731
    splits = MoreHopQADataset(loader=loader).load()
    assert "hf://datasets/alabnii/morehopqa@abc/data/u.json" in urls
    train_ids = {p.problem_id for p in splits["train"]} | {p.problem_id for p in splits["validation"]}
    assert len(train_ids) == 10 and "morehopqa/train/t0" not in train_ids  # t0 shares its first hop with Q0
    assert not any(p.metadata["grading"]["human_verified"] for p in splits["train"])
    assert all(p.problem_id.startswith("morehopqa/test/") for p in splits["test"])


def test_refuses_oversized_samples(morehopqa):
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


def test_grader_falls_back_to_the_answer_line():
    problem = Problem("morehopqa/test/2", "When?", "1880-01-15", "test", {"answer_type": "date"})
    grade = MoreHopQAGrader().grade(
        problem, "REASONING: one day after 14 January 1880\nANSWER: January 15, 1880\nTASK_DONE"
    )
    assert grade.label == 0 and grade.predicted == "January 15, 1880"
