"""`dataset.name` -> (provider, grader). A new dataset adds a subpackage and one entry here."""

from collections.abc import Callable

from spuma.dataset.base import AnswerGrader, DatasetProvider
from spuma.dataset.config import DatasetConfig
from spuma.dataset.math import MathDataset, MathVerifyGrader

DATASETS: dict[str, tuple[Callable[[DatasetConfig], DatasetProvider], Callable[[], AnswerGrader]]] = {
    "math": (MathDataset, MathVerifyGrader),
}


def get_dataset(config: DatasetConfig) -> DatasetProvider:
    return _entry(config.name)[0](config)


def get_grader(config: DatasetConfig) -> AnswerGrader:
    return _entry(config.name)[1]()


def _entry(name: str) -> tuple[Callable[[DatasetConfig], DatasetProvider], Callable[[], AnswerGrader]]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name]
