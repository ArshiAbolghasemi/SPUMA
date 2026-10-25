"""`dataset.name` -> (provider, grader). A new dataset adds a subpackage and one entry here."""

import dataclasses
from collections.abc import Callable
from typing import Any

from spuma.config import get_settings
from spuma.dataset.base import AnswerGrader, DatasetProvider
from spuma.dataset.math import MathDataset, MathVerifyGrader
from spuma.dataset.medhopqa import MedHopQADataset, MedHopQAGrader
from spuma.dataset.mmlu import MMLUDataset, MMLUGrader
from spuma.dataset.morehopqa import MoreHopQADataset, MoreHopQAGrader
from spuma.dataset.musique import MuSiQueDataset, MuSiQueGrader

DATASETS: dict[str, tuple[Callable[[], DatasetProvider], Callable[[], AnswerGrader]]] = {
    "math": (MathDataset, MathVerifyGrader),
    "morehopqa": (MoreHopQADataset, MoreHopQAGrader),
    "mmlu": (MMLUDataset, MMLUGrader),
    "musique": (MuSiQueDataset, MuSiQueGrader),
    "medhopqa": (MedHopQADataset, MedHopQAGrader),
}
ADAPTERS = {"mmlu": MMLUDataset, "musique": MuSiQueDataset, "medhopqa": MedHopQADataset}


def load_dataset_adapter(dataset_name: str, split: str, config: Any = None, seed: int | None = None) -> list[dict]:
    """The standardized examples of one selected split (train / validation / test) of a question-answering dataset:
    {problem_id, dataset, question, choices, context, subject, answer_type, reference_answer, grading_metadata}.
    `config`: a `DatasetConfig` (default: the `dataset:` settings, with `dataset_name`); `seed` overrides its seed.
    The same selection the experiments use, so every architecture sees the same problems."""
    if dataset_name not in ADAPTERS:
        raise ValueError(f"no adapter for {dataset_name!r}; adapters: {sorted(ADAPTERS)}")
    config = config or get_settings().dataset
    if config.name != dataset_name:
        raise ValueError(f"dataset config is for {config.name!r}, not {dataset_name!r}")
    if seed is not None:
        config = dataclasses.replace(config, seed=seed)
    provider = ADAPTERS[dataset_name](config=config)
    selected = {p.problem_id for p in provider.load()[split]}
    sources = {provider.sources[split]}
    examples = [e for source in sources for e in provider.adapter.examples(source) if e.problem_id in selected]
    return [dataclasses.asdict(e) for e in sorted(examples, key=lambda e: e.problem_id)]


def get_dataset() -> DatasetProvider:
    """The provider for `dataset.name`."""
    return _entry(get_settings().dataset.name)[0]()


def get_grader() -> AnswerGrader:
    """The grader for `dataset.name`."""
    return _entry(get_settings().dataset.name)[1]()


def _entry(name: str) -> tuple[Callable[[], DatasetProvider], Callable[[], AnswerGrader]]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name]
