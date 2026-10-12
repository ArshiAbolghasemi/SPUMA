import pytest

from spuma.dataset import Problem, get_dataset, get_grader, make_splits, normalize
from spuma.dataset.base import load_hf_rows
from spuma.dataset.config import DatasetConfig
from spuma.dataset.math import MathDataset, MathVerifyGrader


def _pool(split, n):
    return [Problem(f"{split}/{i}", f"q{i}", f"s{i}", split) for i in range(n)]


def _cfg(**kw):
    return DatasetConfig(**({"train_samples": 5, "validation_samples": 3, "test_samples": 4} | kw))


def test_splits_disjoint_and_respect_official_boundaries():
    splits = make_splits(_pool("train", 10), _pool("test", 6), _cfg())
    assert {k: len(v) for k, v in splits.items()} == {"train": 5, "validation": 3, "test": 4}
    train_ids = {p.problem_id for p in splits["train"]}
    val_ids = {p.problem_id for p in splits["validation"]}
    assert not train_ids & val_ids
    assert all(i.startswith("train/") for i in train_ids | val_ids)
    assert all(p.problem_id.startswith("test/") for p in splits["test"])
    assert {p.split for p in splits["validation"]} == {"validation"}


def test_splits_depend_only_on_seed_and_ids():
    ids = lambda s: [p.problem_id for p in s["train"]]  # noqa: E731
    a = make_splits(_pool("train", 10), _pool("test", 6), _cfg())
    b = make_splits(list(reversed(_pool("train", 10))), _pool("test", 6), _cfg())
    c = make_splits(_pool("train", 10), _pool("test", 6), _cfg(seed=7))
    assert ids(a) == ids(b) and ids(a) != ids(c)


def test_too_many_requested():
    with pytest.raises(ValueError, match="available"):
        make_splits(_pool("train", 10), _pool("test", 6), _cfg(test_samples=99))


def test_load_hf_rows():
    rows = list(load_hf_rows(lambda path, subset, split: [{"x": subset}] * 2, "p", ["a", "b"], "train"))
    assert [(s, i) for s, i, _ in rows] == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]


def test_normalize():
    assert normalize("  The  Answer\n") == "the answer"


def test_registry():
    assert isinstance(get_dataset(DatasetConfig()), MathDataset)
    assert isinstance(get_grader(DatasetConfig()), MathVerifyGrader)
