"""Dataset-agnostic pieces: the problem record, provider/grader interfaces, grades, split construction.

Reference solutions live on `Problem` and are only read by graders; the workflow receives
`problem_id` and `problem` text only.

`DatasetExample` is the standardized record of the question-answering datasets (MMLU, MuSiQue, MoreHopQA):
agent-visible fields (question, choices, context) are rendered into `Problem.problem` by `to_problem`; private
fields (reference answer, grading metadata: aliases, synonyms, decompositions, supporting-paragraph labels) go
only to `Problem.reference_solution` / `Problem.metadata["grading"]`, which graders read and agents never see.
"""

import logging
import random
import re
import string
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Protocol

from spuma.config import get_settings

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    problem: str
    reference_solution: str | None  # None: unlabeled (blind-test question), never graded as right or wrong
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetExample:
    """One question in the standardized form. Agent-visible: question, choices, context. Private: the rest."""

    problem_id: str
    dataset: str
    question: str
    choices: list[str] | None
    context: list[dict[str, Any]] | None  # [{"id": ..., "title": ..., "text": ...}]
    subject: str | None
    answer_type: str
    reference_answer: Any | None
    grading_metadata: dict[str, Any] = field(default_factory=dict)

    def visible_text(self) -> str:
        """What the agents see: the question, then the lettered choices or the numbered context paragraphs (a
        closed-book question alone is shown as it is)."""
        if self.choices is None and self.context is None:
            return self.question
        parts = [f"Question:\n{self.question.strip()}"]
        if self.choices is not None:
            parts.append(
                "Choices:\n"
                + "\n".join(f"{letter}. {c}" for letter, c in zip(string.ascii_uppercase, self.choices, strict=False))
            )
        if self.context is not None:
            paragraphs = [f"[{p['id']}] {p['title']}\n{p['text'].strip()}" for p in self.context]
            parts.append("Context paragraphs:\n\n" + "\n\n".join(paragraphs))
        return "\n\n".join(parts)

    def to_problem(self, split: str) -> Problem:
        reference = None if self.reference_answer is None else str(self.reference_answer)
        metadata = {
            "dataset": self.dataset,
            "subject": self.subject,
            "answer_type": self.answer_type,
            "grading": self.grading_metadata,
        }
        return Problem(self.problem_id, self.visible_text(), reference, split, metadata)


class DatasetProvider(Protocol):
    def load(self) -> dict[str, list[Problem]]:
        """{'train': [...], 'validation': [...], 'test': [...]}"""
        ...


@dataclass(frozen=True)
class Grade:
    problem_id: str
    predicted: str | None
    reference: str | None
    correct: bool | None  # None: unlabeled problem
    label: int | None  # 1 = wrong; None: unlabeled problem (no reference answer)
    parse_failure: bool
    reference_parse_failure: bool
    method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetAdapter(Protocol):
    def examples(self, official_split: str) -> list[DatasetExample]:
        """Every example of one official split of the source (sorted order is irrelevant: splits re-sort by id)."""
        ...


class AnswerGrader(Protocol):
    def grade(self, problem: Problem, final_text: str) -> Grade: ...


def normalize(text: str) -> str:
    """Whitespace-collapsed, lower-cased text."""
    return " ".join(text.split()).lower()


ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization: lower case, no punctuation or articles, collapsed whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(ARTICLES_RE.sub(" ", text).split())


def load_hf_rows(
    loader: Callable[..., Sequence[dict[str, Any]]], path: str, subsets: list[str], split: str
) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """(subset, row index, row) for every row of every subset of one official split of a HF dataset."""
    for subset in subsets:
        for i, row in enumerate(loader(path, subset, split=split)):
            yield subset, i, row


def make_splits(train_pool: list[Problem], test_pool: list[Problem]) -> dict[str, list[Problem]]:
    """Seeded splits that keep official boundaries: train and validation are disjoint draws from the official
    train pool, test is drawn from the official test pool. Pools are sorted by id first, so the result depends
    only on the seed and the problem ids. Sizes and seed come from the `dataset:` settings."""
    return select_splits(
        {"train": "train", "validation": "train", "test": "test"}, {"train": train_pool, "test": test_pool}
    )


def select_splits(
    sources: dict[str, str], pools: dict[str, list[Problem]], config: Any = None
) -> dict[str, list[Problem]]:
    """train / validation / test drawn from official pools: `sources` maps each to its official split. Each pool is
    sorted by id and shuffled once with "<seed>:<pool>"; the splits that share a pool take consecutive, disjoint
    slices in train, validation, test order, so no problem is in two splits. `config`: a `DatasetConfig` (default:
    the `dataset:` settings)."""
    config = config or get_settings().dataset
    sizes = {"train": config.train_samples, "validation": config.validation_samples, "test": config.test_samples}
    shuffled = {name: _shuffled(pool, f"{config.seed}:{name}") for name, pool in pools.items()}
    taken = dict.fromkeys(pools, 0)
    out = {}
    for split in SPLITS:
        source = sources[split]
        pool, start = shuffled[source], taken[source]
        if start + sizes[split] > len(pool):
            raise ValueError(
                f"requested {sizes[split]} {split} problems from the official {source!r} split after {start} taken "
                f"by earlier splits; available {len(pool)}"
            )
        out[split] = [replace(p, split=split) for p in pool[start : start + sizes[split]]]
        taken[source] = start + sizes[split]
    return out


class AdapterDataset:
    """`DatasetProvider` over a `DatasetAdapter`: the official splits named by `dataset.split_sources` (default:
    the adapter's `SPLIT_SOURCES`) become train / validation / test via `select_splits`."""

    def __init__(self, adapter: DatasetAdapter, default_sources: dict[str, str], config: Any = None) -> None:
        self.adapter = adapter
        self.config = config or get_settings().dataset
        self.sources = {**default_sources, **self.config.split_sources}

    def load(self) -> dict[str, list[Problem]]:
        sizes = self.config
        needed = {
            self.sources[s]
            for s, n in (
                ("train", sizes.train_samples),
                ("validation", sizes.validation_samples),
                ("test", sizes.test_samples),
            )
            if n
        }
        pools = {
            name: [e.to_problem(name) for e in self.adapter.examples(name)] if name in needed else []
            for name in set(self.sources.values())
        }
        splits = select_splits(self.sources, pools, self.config)
        logger.info(
            "dataset splits",
            extra={
                "dataset": self.config.name,
                "sources": self.sources,
                "sizes": {k: len(v) for k, v in splits.items()},
            },
        )
        return splits


def _shuffled(pool: list[Problem], seed: str) -> list[Problem]:
    out = sorted(pool, key=lambda p: p.problem_id)
    random.Random(seed).shuffle(out)
    return out
