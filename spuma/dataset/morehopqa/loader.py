"""MoreHopQA human-verified test set from the Hugging Face Hub, read as parquet at a pinned revision.

The Hub repository ships a loading script that current `datasets` no longer runs, so the verified parquet
file (`dataset.data_files`) is read directly. The set is closed-book (no context paragraphs): the agents
see only the question. There is no train split; `dataset.test_samples` problems are drawn with
`dataset.seed`. Problem ids are stable: `morehopqa/test/<row>`.
"""

import logging
from typing import Any

from datasets import load_dataset

from spuma.config import get_settings
from spuma.dataset.base import Problem, _shuffled

logger = logging.getLogger(__name__)


def _load_parquet(url: str) -> list[dict[str, Any]]:
    return list(load_dataset("parquet", data_files=url, split="train"))


class MoreHopQADataset:
    """Reads the `dataset:` settings; `loader` replaces the parquet reader (tests)."""

    def __init__(self, loader: Any = None) -> None:
        self.config = get_settings().dataset
        self.loader = loader or _load_parquet

    def url(self) -> str:
        c = self.config
        if not c.data_files:
            raise ValueError("dataset.data_files must name the MoreHopQA parquet file")
        revision = f"@{c.revision}" if c.revision else ""
        return f"hf://datasets/{c.hf_path}{revision}/{c.data_files}"

    def load(self) -> dict[str, list[Problem]]:
        c = self.config
        if c.train_samples or c.validation_samples:
            raise ValueError("MoreHopQA is an evaluation-only dataset: set train_samples and validation_samples to 0")
        rows = self.loader(self.url())
        pool = [
            Problem(
                problem_id=f"morehopqa/test/{i}",
                problem=row["question"],
                reference_solution=str(row["answer"]),
                split="test",
                metadata={
                    "answer_type": row.get("answer_type"),
                    "no_of_hops": row.get("no_of_hops"),
                    "reasoning_type": row.get("reasoning_type"),
                },
            )
            for i, row in enumerate(rows)
        ]
        if c.test_samples > len(pool):
            raise ValueError(f"requested {c.test_samples} test problems; MoreHopQA has {len(pool)}")
        test = _shuffled(pool, f"{c.seed}:test")[: c.test_samples]
        logger.info("MoreHopQA test sample", extra={"size": len(test), "pool": len(pool), "seed": c.seed})
        return {"train": [], "validation": [], "test": test}
