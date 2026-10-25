"""MMLU from the Hugging Face Hub (`cais/mmlu`, the `all` config's parquet files at `dataset.revision`).

Official splits: dev (285, 5 per subject), validation (1,531), test (14,042), auxiliary_train (99,842; other
multiple-choice sets: ARC, OBQA, RACE, MCTest). MMLU has no in-distribution training split, so the default
`split_sources` is explicit and documented here and in each settings file:
    train      <- official validation   (SPN, CHMM and calibrator fitting)
    validation <- official dev
    test       <- official test         (never used for fitting)
Each target draws from its own official split, so no question is in two splits. `dataset.subsets` limits the
subjects (empty or ["all"]: all 57). The agents see the question and the four lettered choices; the gold letter
is private. Problem ids: `mmlu/<official split>/<subject>/<row>`.
"""

import logging
from typing import Any

from datasets import load_dataset

from spuma.config import get_settings
from spuma.dataset.base import AdapterDataset, DatasetExample, Problem
from spuma.dataset.qa import QAGrader, option_letter

logger = logging.getLogger(__name__)

SPLIT_SOURCES = {"train": "validation", "validation": "dev", "test": "test"}
LETTERS = "ABCD"


def _load_parquet(url: str) -> list[dict[str, Any]]:
    return list(load_dataset("parquet", data_files=url, split="train"))


class MMLUAdapter:
    def __init__(self, config: Any, loader: Any = None) -> None:
        self.config, self.loader = config, loader or _load_parquet

    def url(self, official_split: str) -> str:
        c = self.config
        revision = f"@{c.revision}" if c.revision else ""
        return f"hf://datasets/{c.hf_path}{revision}/all/{official_split}-00000-of-00001.parquet"

    def examples(self, official_split: str) -> list[DatasetExample]:
        subjects = {s for s in self.config.subsets if s != "all"}
        out = []
        for i, row in enumerate(self.loader(self.url(official_split))):
            subject = row.get("subject") or "unknown"
            if subjects and subject not in subjects:
                continue
            if len(row["choices"]) != 4:
                raise ValueError(f"MMLU {official_split} row {i}: expected 4 choices, got {len(row['choices'])}")
            out.append(
                DatasetExample(
                    problem_id=f"mmlu/{official_split}/{subject}/{i}",
                    dataset="mmlu",
                    question=row["question"],
                    choices=[str(c) for c in row["choices"]],
                    context=None,
                    subject=subject,
                    answer_type="option_letter",
                    reference_answer=LETTERS[int(row["answer"])],
                    grading_metadata={"source_split": official_split, "row": i},
                )
            )
        logger.info("MMLU split loaded", extra={"split": official_split, "examples": len(out)})
        return out


class MMLUDataset(AdapterDataset):
    def __init__(self, loader: Any = None, config: Any = None) -> None:
        config = config or get_settings().dataset
        super().__init__(MMLUAdapter(config, loader), SPLIT_SOURCES, config)


def _match(predicted: str, problem: Problem) -> tuple[bool, str]:
    letter = option_letter(predicted)
    return letter is not None and letter == problem.reference_solution, "option_letter"


class MMLUGrader(QAGrader):
    """Correct iff the final `ANSWER:` is the gold option letter (an unparseable letter is wrong)."""

    def __init__(self) -> None:
        super().__init__(_match)
