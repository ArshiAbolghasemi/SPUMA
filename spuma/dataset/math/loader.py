"""MATH benchmark (Hendrycks et al.) from the Hugging Face Hub (`dataset.hf_path`, one config per subject).

Problem ids are stable: `math/<official split>/<subject>/<row>`.
"""

import logging
from typing import Any

from datasets import load_dataset

from spuma.config import get_settings
from spuma.dataset.base import Problem, load_hf_rows, make_splits

logger = logging.getLogger(__name__)


class MathDataset:
    """Reads the `dataset:` settings; `loader` replaces `datasets.load_dataset` (tests)."""

    def __init__(self, loader: Any = None) -> None:
        self.config = get_settings().dataset
        if loader is None:
            loader = load_dataset
        self.loader = loader

    def _pool(self, official_split: str) -> list[Problem]:
        return [
            Problem(
                problem_id=f"math/{official_split}/{subset}/{i}",
                problem=row["problem"],
                reference_solution=row["solution"],
                split=official_split,
                metadata={"subject": row.get("type", subset), "level": str(row.get("level", ""))},
            )
            for subset, i, row in load_hf_rows(self.loader, self.config.hf_path, self.config.subsets, official_split)
        ]

    def load(self) -> dict[str, list[Problem]]:
        splits = make_splits(self._pool("train"), self._pool("test"))
        logger.info("MATH splits", extra={"sizes": {k: len(v) for k, v in splits.items()}, "seed": self.config.seed})
        return splits
