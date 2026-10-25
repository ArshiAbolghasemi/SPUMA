"""MuSiQue v1.0 JSONL (`dataset.variant`: ans | full), one file per official split (`dataset.data_files`).

MuSiQue-Ans default: the verified Hub mirror `dgslibisey/MuSiQue` at a pinned revision, whose train/dev files
match the official v1.0 release (19,938 / 2,417 questions, the official schema). MuSiQue-Full (answerable and
unanswerable contrast pairs) is only in the official release (github.com/StonyBrookNLP/musique, `musique_v1.0.zip`):
set `data_files` to its `musique_full_v1.0_{train,dev}.jsonl`, local paths with `source: local`.

The official test files have no answers, so the default `split_sources` keeps every split labeled:
    train, validation <- official train (disjoint seeded draws)
    test              <- official dev
The agents see the question and all 20 paragraphs ("[idx] title", supporting and distractor alike). Private,
for grading and analysis only: answer, aliases, answerability, the gold decomposition and the supporting-paragraph
labels. Problem ids: `musique_<variant>/<official split>/<id>`.
"""

import logging
from typing import Any

from datasets import load_dataset

from spuma.config import get_settings
from spuma.dataset.base import AdapterDataset, DatasetExample, Problem
from spuma.dataset.morehopqa.grading import normalize_answer
from spuma.dataset.qa import INSUFFICIENT, QAGrader, is_insufficient

logger = logging.getLogger(__name__)

SPLIT_SOURCES = {"train": "train", "validation": "train", "test": "dev"}
MIRROR_FILES = {"train": "musique_ans_v1.0_train.jsonl", "dev": "musique_ans_v1.0_dev.jsonl"}
REQUIRED = ("id", "paragraphs", "question")


def _load_json(path: str) -> list[dict[str, Any]]:
    return list(load_dataset("json", data_files=path, split="train"))


class MuSiQueAdapter:
    def __init__(self, config: Any, loader: Any = None) -> None:
        self.config, self.loader = config, loader or _load_json

    def path(self, official_split: str) -> str:
        c = self.config
        files = c.data_files if isinstance(c.data_files, dict) else {}
        if official_split in files:
            return files[official_split]
        if c.variant == "ans" and c.source == "huggingface" and official_split in MIRROR_FILES:
            revision = f"@{c.revision}" if c.revision else ""
            return f"hf://datasets/{c.hf_path}{revision}/{MIRROR_FILES[official_split]}"
        raise ValueError(
            f"dataset.data_files names no file for the official MuSiQue-{c.variant} {official_split!r} split"
        )

    def examples(self, official_split: str) -> list[DatasetExample]:
        variant = self.config.variant
        out = []
        for row in self.loader(self.path(official_split)):
            missing = [k for k in REQUIRED if k not in row]
            if missing:
                raise ValueError(f"MuSiQue row {row.get('id')!r} lacks {missing}; fields: {sorted(row)}")
            answerable = bool(row.get("answerable", True))
            labeled = "answer" in row and row["answer"] is not None
            reference = (row["answer"] if answerable else INSUFFICIENT) if labeled else None
            out.append(
                DatasetExample(
                    problem_id=f"musique_{variant}/{official_split}/{row['id']}",
                    dataset="musique",
                    question=row["question"],
                    choices=None,
                    context=[
                        {"id": p["idx"], "title": p["title"], "text": p["paragraph_text"]} for p in row["paragraphs"]
                    ],
                    subject=None,
                    answer_type="short_answer" if answerable else "unanswerable",
                    reference_answer=reference,
                    grading_metadata={
                        "variant": variant,
                        "source_split": official_split,
                        "answer": row.get("answer"),
                        "answer_aliases": list(row.get("answer_aliases") or []),
                        "answerable": answerable,
                        "hops": len(row.get("question_decomposition") or []),
                        "supporting_paragraphs": [p["idx"] for p in row["paragraphs"] if p.get("is_supporting")],
                        "question_decomposition": row.get("question_decomposition"),
                    },
                )
            )
        logger.info("MuSiQue split loaded", extra={"variant": variant, "split": official_split, "examples": len(out)})
        return out


class MuSiQueDataset(AdapterDataset):
    def __init__(self, loader: Any = None, config: Any = None) -> None:
        config = config or get_settings().dataset
        super().__init__(MuSiQueAdapter(config, loader), SPLIT_SOURCES, config)


def _match(predicted: str, problem: Problem) -> tuple[bool, str]:
    grading = problem.metadata["grading"]
    if not grading["answerable"]:
        return is_insufficient(predicted), "unanswerable"
    if is_insufficient(predicted):
        return False, "insufficient_on_answerable"
    golds = [grading["answer"], *grading["answer_aliases"]]
    return normalize_answer(predicted) in {normalize_answer(g) for g in golds}, "exact_alias"


class MuSiQueGrader(QAGrader):
    """Answerable: SQuAD-normalized exact match against the answer or an alias. Unanswerable (Full): correct
    iff INSUFFICIENT_INFORMATION."""

    def __init__(self) -> None:
        super().__init__(_match)
