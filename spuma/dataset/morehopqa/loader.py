"""MoreHopQA (`alabnii/morehopqa` on the Hugging Face Hub, at a pinned `dataset.revision`), closed-book: the agents
see only the question (the context paragraphs are not shown).

Official splits:
    test   human-verified set, 1,118 questions (`verified/test-00000-of-00001.parquet`); ids `morehopqa/test/<row>`
    train  `data/without_human_verification.json`, 2,502 generated questions whose answers were NOT human-verified;
           ids `morehopqa/train/<_id>`. None shares an id or question with the verified set, but some are built on
           the same first-hop HotpotQA question (`previous_question`) as a verified one; those are dropped, so no
           training question shares its first hop with any verified question.
The default `split_sources` keeps the verified set for testing only: train and validation are disjoint draws from
`train`, test from `test`. `dataset.data_files`: the test parquet (a string, as in the OOD settings), or
{test: ..., train: ...}. The repository's loading script is not run; the files are read directly.
"""

import json
import logging
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from spuma.config import get_settings
from spuma.dataset.base import AdapterDataset, DatasetExample

logger = logging.getLogger(__name__)

SPLIT_SOURCES = {"train": "train", "validation": "train", "test": "test"}
DEFAULT_FILES = {"test": "verified/test-00000-of-00001.parquet", "train": "data/without_human_verification.json"}


def _load_parquet(url: str) -> list[dict[str, Any]]:
    return list(load_dataset("parquet", data_files=url, split="train"))


def _read(url: str) -> list[dict[str, Any]]:
    """Rows of a parquet file, or of a JSON list (`hf://datasets/<repo>@<revision>/<path>` or a local path)."""
    if not url.endswith(".json"):
        return _load_parquet(url)
    if url.startswith("hf://datasets/"):
        owner, name_revision, path = url.removeprefix("hf://datasets/").split("/", 2)
        name, _, revision = name_revision.partition("@")
        url = hf_hub_download(f"{owner}/{name}", path, repo_type="dataset", revision=revision or None)
    with open(url) as f:
        return json.load(f)


def _first_hop(question: str | None) -> str:
    return " ".join((question or "").lower().split())


class MoreHopQAAdapter:
    def __init__(self, config: Any, loader: Any = None) -> None:
        self.config, self.loader = config, loader or _read

    def url(self, official_split: str = "test") -> str:
        c = self.config
        files = c.data_files if isinstance(c.data_files, dict) else {"test": c.data_files} if c.data_files else {}
        path = files.get(official_split) or DEFAULT_FILES[official_split]
        if path.startswith(("hf://", "/")) or c.source == "local":
            return path
        revision = f"@{c.revision}" if c.revision else ""
        return f"hf://datasets/{c.hf_path}{revision}/{path}"

    def examples(self, official_split: str) -> list[DatasetExample]:
        rows = self.loader(self.url(official_split))
        if official_split == "train":
            verified = {_first_hop(r.get("previous_question")) for r in self.loader(self.url("test"))}
            kept = [r for r in rows if _first_hop(r.get("previous_question")) not in verified]
            logger.info("MoreHopQA train: shared first hops dropped", extra={"rows": len(rows), "kept": len(kept)})
            ids = [f"morehopqa/train/{r['_id']}" for r in kept]
            rows = kept
        else:
            ids = [f"morehopqa/test/{i}" for i in range(len(rows))]
        return [
            DatasetExample(
                problem_id=pid,
                dataset="morehopqa",
                question=row["question"],
                choices=None,
                context=None,
                subject=row.get("reasoning_type"),
                answer_type=row.get("answer_type") or "string",
                reference_answer=str(row["answer"]),
                grading_metadata={
                    "source_split": official_split,
                    "human_verified": official_split == "test",
                    "no_of_hops": row.get("no_of_hops"),
                    "reasoning_type": row.get("reasoning_type"),
                },
            )
            for pid, row in zip(ids, rows, strict=True)
        ]


class MoreHopQADataset(AdapterDataset):
    """Reads the `dataset:` settings; `loader` replaces the file reader (tests)."""

    def __init__(self, loader: Any = None, config: Any = None) -> None:
        config = config or get_settings().dataset
        super().__init__(MoreHopQAAdapter(config, loader), SPLIT_SOURCES, config)
