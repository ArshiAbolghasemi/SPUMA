"""Dataset-agnostic pieces: the problem record, provider/grader interfaces, grades, split construction.

Reference solutions live on `Problem` and are only read by graders; the workflow receives
`problem_id` and `problem` text only.
"""

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from spuma.dataset.config import DatasetConfig

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    problem: str
    reference_solution: str
    split: str  # train | validation | test
    metadata: dict[str, Any] = field(default_factory=dict)  # dataset-specific (subject, level, ...)


class DatasetProvider(Protocol):
    def load(self) -> dict[str, list[Problem]]:
        """{'train': [...], 'validation': [...], 'test': [...]}"""
        ...


@dataclass(frozen=True)
class Grade:
    problem_id: str
    predicted: str | None
    reference: str | None
    correct: bool
    label: int  # 1 = incorrect
    parse_failure: bool  # no extractable final answer
    reference_parse_failure: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnswerGrader(Protocol):
    def grade(self, problem: Problem, final_text: str) -> Grade: ...


def normalize(text: str) -> str:
    """Whitespace-collapsed, lower-cased text."""
    return " ".join(text.split()).lower()


def load_hf_rows(
    loader: Callable[..., Sequence[dict[str, Any]]], path: str, subsets: list[str], split: str
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """(subset, row index, row) for every row of every subset of one official split of a HF dataset."""
    for subset in subsets:
        for i, row in enumerate(loader(path, subset, split=split)):
            yield subset, i, row


def make_splits(train_pool: list[Problem], test_pool: list[Problem], config: DatasetConfig) -> dict[str, list[Problem]]:
    """Seeded splits that keep official boundaries: train and validation are disjoint draws from the official
    train pool, test is drawn from the official test pool. Pools are sorted by id first, so the result depends
    only on the seed and the problem ids."""
    need, have = (
        (config.train_samples + config.validation_samples, config.test_samples),
        (len(train_pool), len(test_pool)),
    )
    if need[0] > have[0] or need[1] > have[1]:
        raise ValueError(
            f"requested {config.train_samples}+{config.validation_samples} train/validation and {config.test_samples} "
            f"test problems; available {have[0]} train and {have[1]} test"
        )
    train = _shuffled(train_pool, f"{config.seed}:train")
    test = _shuffled(test_pool, f"{config.seed}:test")
    picks = {
        "train": train[: config.train_samples],
        "validation": train[config.train_samples : config.train_samples + config.validation_samples],
        "test": test[: config.test_samples],
    }
    return {name: [replace(p, split=name) for p in problems] for name, problems in picks.items()}


def _shuffled(pool: list[Problem], seed: str) -> list[Problem]:
    out = sorted(pool, key=lambda p: p.problem_id)
    random.Random(seed).shuffle(out)
    return out
