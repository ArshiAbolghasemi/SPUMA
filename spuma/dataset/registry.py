"""`dataset.name` -> (provider, grader). A new dataset adds a subpackage and one entry here."""

from collections.abc import Callable

from spuma.config import get_settings
from spuma.dataset.base import AnswerGrader, DatasetProvider
from spuma.dataset.math import MathDataset, MathVerifyGrader

DATASETS: dict[str, tuple[Callable[[], DatasetProvider], Callable[[], AnswerGrader]]] = {
    "math": (MathDataset, MathVerifyGrader),
}


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
