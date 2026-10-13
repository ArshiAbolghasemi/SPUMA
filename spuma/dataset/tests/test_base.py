import dataclasses

import pytest

from spuma.dataset.base import Problem, load_hf_rows, make_splits, normalize
from spuma.dataset.math import MathDataset, MathVerifyGrader
from spuma.dataset.registry import get_dataset, get_grader


def _pool(split, n):
    return [Problem(f"{split}/{i}", f"q{i}", f"s{i}", split) for i in range(n)]


@pytest.fixture
def sizes(config, set_settings):
    """Install `dataset:` settings with small split sizes (plus any overrides) for this test."""

    def install(**kw):
        dataset = dataclasses.replace(
            config.dataset, **({"train_samples": 5, "validation_samples": 3, "test_samples": 4} | kw)
        )
        set_settings(dataclasses.replace(config, dataset=dataset))

    install()
    return install


def test_splits_disjoint_and_respect_official_boundaries(sizes):
    splits = make_splits(_pool("train", 10), _pool("test", 6))
    assert {k: len(v) for k, v in splits.items()} == {"train": 5, "validation": 3, "test": 4}
    train_ids = {p.problem_id for p in splits["train"]}
    val_ids = {p.problem_id for p in splits["validation"]}
    assert not train_ids & val_ids
    assert all(i.startswith("train/") for i in train_ids | val_ids)
    assert all(p.problem_id.startswith("test/") for p in splits["test"])
    assert {p.split for p in splits["validation"]} == {"validation"}


def test_splits_depend_only_on_seed_and_ids(sizes):
    ids = lambda s: [p.problem_id for p in s["train"]]  # noqa: E731
    a = make_splits(_pool("train", 10), _pool("test", 6))
    b = make_splits(list(reversed(_pool("train", 10))), _pool("test", 6))
    sizes(seed=7)
    c = make_splits(_pool("train", 10), _pool("test", 6))
    assert ids(a) == ids(b) and ids(a) != ids(c)


def test_too_many_requested(sizes):
    sizes(test_samples=99)
    with pytest.raises(ValueError, match="available"):
        make_splits(_pool("train", 10), _pool("test", 6))


def test_load_hf_rows():
    rows = list(load_hf_rows(lambda path, subset, split: [{"x": subset}] * 2, "p", ["a", "b"], "train"))
    assert [(s, i) for s, i, _ in rows] == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]


def test_normalize():
    assert normalize("  The  Answer\n") == "the answer"


def test_registry():
    assert isinstance(get_dataset(), MathDataset)  # dataset.name: math
    assert isinstance(get_grader(), MathVerifyGrader)
