"""MMLU, MuSiQue and MedHopQA: adapters, split selection, agent-visible vs private fields, answer extraction and
grading. Fake loaders stand in for the Hub / local files (offline)."""

import dataclasses
import json

import pytest

from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, get_settings
from spuma.dataset.base import Problem
from spuma.dataset.medhopqa import MedHopQADataset, MedHopQAGrader
from spuma.dataset.medhopqa.loader import detect_fields
from spuma.dataset.mmlu import MMLUDataset, MMLUGrader
from spuma.dataset.musique import MuSiQueDataset, MuSiQueGrader
from spuma.dataset.qa import any_answer, final_answer, is_insufficient, option_letter
from spuma.dataset.registry import load_dataset_adapter

ARCHITECTURES = ("feedbackloop", "star", "sequential")


def _settings(monkeypatch, dataset, architecture="sequential", **dataset_overrides):
    path = PROJECT_ROOT / "config" / architecture / f"{dataset}_ma_qwen2_5_7b_{architecture}.yaml"
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    get_settings.cache_clear()
    s = get_settings()
    return dataclasses.replace(s.dataset, **dataset_overrides) if dataset_overrides else s.dataset


def mmlu_loader(url):
    split = url.split("/all/")[1].split("-")[0]
    n = {"dev": 20, "validation": 40, "test": 60}[split]
    return [
        {
            "question": f"{split} question {i}?",
            "subject": "anatomy" if i % 2 else "law",
            "choices": ["w", "x", "y", "z"],
            "answer": i % 4,
        }
        for i in range(n)
    ]


def musique_rows(split, n, answerable=True, labeled=True):
    rows = []
    for i in range(n):
        row = {
            "id": f"2hop__{split}_{i}",
            "question": f"Who founded the club in question {i}?",
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Club",
                    "paragraph_text": "The club was founded by its patron.",
                    "is_supporting": True,
                },
                {"idx": 1, "title": "Distractor", "paragraph_text": "Unrelated text.", "is_supporting": False},
            ],
            "question_decomposition": [{"id": 1, "question": "SECRET_DECOMPOSITION club >> founder", "answer": "Zed"}],
            "answerable": answerable,
        }
        if labeled:
            row |= {"answer": "Zed Zorin", "answer_aliases": ["Z. Zorin"]}
        rows.append(row)
    return rows


def test_answer_extraction():
    text = "REASONING: x\nVERIFIED_ANSWER: C\n**ANSWER:** B. Paris\nTASK_DONE"
    assert final_answer(text) == "B. Paris" and any_answer(text) == "B. Paris"
    assert final_answer("VERDICT: CORRECT\nVERIFIED_ANSWER: 42") is None and any_answer("SUGGESTED_ANSWER: 42") == "42"
    assert final_answer("ANSWER: first\nmore\nANSWER: Zed Zorin TASK_DONE") == "Zed Zorin"
    letters = ("B", "(c)", "D. 4", "Option A", "C mitochondria", "E", "Paris")
    assert [option_letter(a) for a in letters] == ["B", "C", "D", "A", "C", None, None]
    assert is_insufficient("INSUFFICIENT_INFORMATION") and is_insufficient("insufficient information")
    assert not is_insufficient("Zed")


def test_mmlu_splits_come_from_their_own_official_splits_and_hide_the_gold_letter(monkeypatch):
    config = _settings(monkeypatch, "mmlu", train_samples=10, validation_samples=5, test_samples=8)
    splits = MMLUDataset(mmlu_loader, config).load()
    assert {k: len(v) for k, v in splits.items()} == {"train": 10, "validation": 5, "test": 8}
    assert {p.problem_id.split("/")[1] for p in splits["train"]} == {"validation"}
    assert {p.problem_id.split("/")[1] for p in splits["validation"]} == {"dev"}
    assert {p.problem_id.split("/")[1] for p in splits["test"]} == {"test"}
    p = splits["test"][0]
    assert "A. w\nB. x\nC. y\nD. z" in p.problem and p.reference_solution in "ABCD"
    assert "answer" not in p.problem.lower() and p.metadata["subject"] in ("anatomy", "law")
    again = MMLUDataset(mmlu_loader, config).load()
    assert [x.problem_id for x in again["test"]] == [x.problem_id for x in splits["test"]]
    law_only = MMLUDataset(mmlu_loader, dataclasses.replace(config, subsets=["law"])).load()
    assert {x.metadata["subject"] for s in law_only.values() for x in s} == {"law"}


def test_mmlu_grading():
    problem = Problem("mmlu/test/law/0", "q", "B", "test", {"grading": {}})
    grader = MMLUGrader()
    assert grader.grade(problem, "REASONING: r\nANSWER: B\nTASK_DONE").label == 0
    assert grader.grade(problem, "ANSWER: (C)").label == 1
    no_answer = grader.grade(problem, "I think B")
    assert no_answer.label == 1 and no_answer.parse_failure


def test_musique_keeps_supporting_labels_and_decompositions_private(monkeypatch):
    config = _settings(monkeypatch, "musique", train_samples=3, validation_samples=2, test_samples=4)
    loader = lambda path: musique_rows("train" if "train" in path else "dev", 6)  # noqa: E731
    splits = MuSiQueDataset(loader, config).load()
    train_ids = {p.problem_id for p in splits["train"]} | {p.problem_id for p in splits["validation"]}
    assert len(train_ids) == 5 and all("/train/" in i for i in train_ids)
    assert all("/dev/" in p.problem_id for p in splits["test"])
    p = splits["test"][0]
    assert "[0] Club" in p.problem and "[1] Distractor" in p.problem and "Unrelated text." in p.problem
    for secret in ("is_supporting", "SECRET_DECOMPOSITION", "Zed Zorin", "Z. Zorin", "answerable"):
        assert secret not in p.problem
    assert p.metadata["grading"]["supporting_paragraphs"] == [0]


def test_musique_grading_aliases_and_unanswerable(monkeypatch):
    config = _settings(monkeypatch, "musique", train_samples=0, validation_samples=0, test_samples=2)
    grader = MuSiQueGrader()
    (answerable,) = MuSiQueDataset(
        lambda _: musique_rows("dev", 1), dataclasses.replace(config, test_samples=1)
    ).load()["test"]
    assert grader.grade(answerable, "EVIDENCE: [0]\nANSWER: z. zorin").label == 0
    assert grader.grade(answerable, "ANSWER: INSUFFICIENT_INFORMATION").label == 1
    full = dataclasses.replace(config, variant="full", data_files={"dev": "full_dev.jsonl"}, test_samples=1)
    (unanswerable,) = MuSiQueDataset(lambda _: musique_rows("dev", 1, answerable=False), full).load()["test"]
    assert (
        unanswerable.problem_id.startswith("musique_full/")
        and unanswerable.reference_solution == "INSUFFICIENT_INFORMATION"
    )
    assert grader.grade(unanswerable, "ANSWER: INSUFFICIENT_INFORMATION").label == 0
    assert grader.grade(unanswerable, "ANSWER: Zed Zorin").label == 1
    (blind,) = MuSiQueDataset(
        lambda _: musique_rows("dev", 1, labeled=False), dataclasses.replace(config, test_samples=1)
    ).load()["test"]
    assert blind.reference_solution is None


def test_medhopqa_detects_columns_keeps_only_labeled_and_grades_synonyms(monkeypatch):
    config = _settings(monkeypatch, "medhopqa")
    rows = [
        {"qid": "q1", "Question": "Which gene is mutated in X?", "answer": ["BRCA1", "breast cancer 1"]},
        {
            "qid": "q2",
            "Question": "Which organism causes Y?",
            "answer": "Plasmodium falciparum",
            "synonyms": "P. falciparum",
        },
        {"qid": "q3", "Question": "Blind question?", "answer": None},
    ]
    splits = MedHopQADataset(lambda _: rows, config).load()
    test = {p.problem_id: p for p in splits["test"]}
    assert set(test) == {"medhopqa/dev/q1", "medhopqa/dev/q2"} and not splits["train"]
    q1, q2 = test["medhopqa/dev/q1"], test["medhopqa/dev/q2"]
    assert "BRCA1" not in q1.problem and q1.problem.startswith("Question:\nWhich gene")
    grader = MedHopQAGrader()
    assert grader.grade(q1, "ANSWER: Breast Cancer 1").label == 0
    assert grader.grade(q2, "REASONING: r\nANSWER: P. falciparum\nTASK_DONE").label == 0
    assert grader.grade(q2, "ANSWER: Plasmodium vivax").label == 1
    all_rows = MedHopQADataset(lambda _: rows, dataclasses.replace(config, labeled_only=False)).load()["test"]
    assert sum(p.reference_solution is None for p in all_rows) == 1


def test_medhopqa_schema_problems_name_the_actual_columns(monkeypatch):
    with pytest.raises(ValueError, match="ambiguous MedHopQA id column"):
        detect_fields(["id", "qid", "question"], {})
    with pytest.raises(ValueError, match=r"no MedHopQA question column.*\['id', 'text'\]"):
        detect_fields(["id", "text"], {})
    assert detect_fields(["id", "text"], {"question": "text"})["question"] == "text"
    config = _settings(monkeypatch, "medhopqa")
    with pytest.raises(ValueError, match="no labeled training data"):
        MedHopQADataset(lambda _: [], dataclasses.replace(config, train_samples=5))


@pytest.mark.parametrize("dataset", ["mmlu", "musique", "medhopqa"])
def test_every_architecture_selects_the_same_problems(monkeypatch, dataset):
    sections = []
    for architecture in ARCHITECTURES:
        _settings(monkeypatch, dataset, architecture)
        sections.append((get_settings().dataset, get_settings().project.seed))
    assert sections[0] == sections[1] == sections[2]


def test_load_dataset_adapter_returns_standardized_examples(monkeypatch):
    config = _settings(monkeypatch, "mmlu", train_samples=4, validation_samples=2, test_samples=3)
    import spuma.dataset.mmlu.loader as mmlu  # noqa: PLC0415

    monkeypatch.setattr(mmlu, "_load_parquet", mmlu_loader)
    test = load_dataset_adapter("mmlu", "test", config, seed=7)
    assert len(test) == 3 and set(test[0]) == {
        "problem_id",
        "dataset",
        "question",
        "choices",
        "context",
        "subject",
        "answer_type",
        "reference_answer",
        "grading_metadata",
    }
    assert test == load_dataset_adapter("mmlu", "test", config, seed=7)
    assert json.dumps(test) and all(e["dataset"] == "mmlu" and len(e["choices"]) == 4 for e in test)
