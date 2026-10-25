"""MedHopQA from local files (`source: local`, `dataset.data_files: {official split name: path}`).

The data is distributed through the BioCreative IX CodaBench competition (codabench.org/competitions/7609;
registration required; description: Zenodo 10.5281/zenodo.21797620, CC BY 4.0): a development set with gold
answers (~45 questions) and a ~10,000-question package with answers withheld, in which the 1,000 scored test
questions are hidden. There is no labeled training data, so:
  - questions with a gold answer are labeled; the others are unlabeled (`reference_solution = None`, grade
    `label = None`): they can be generated and scored but never count as right or wrong;
  - `labeled_only: true` (default) keeps only labeled questions; supervised SPN/CHMM training on MedHopQA is
    not possible (train_samples and validation_samples must be 0), so its settings use the MATH-trained models,
    frozen (`chmm.frozen_from`).
Files: JSON (list or {"data": [...]}), JSONL, CSV or TSV. The schema is not documented, so columns are detected
from the file (exactly one candidate per field) or set in `dataset.fields`, and the mapping is logged; a missing
or ambiguous column fails with the file's actual columns. Answers may be a string or a list (first = gold, rest =
synonyms); synonyms a list or a "|"/";"-separated string. Agents see the question only: the Wikipedia source
pages are not part of the release, and no retrieval is configured. Problem ids: `medhopqa/<split>/<id>`.

Grading follows the official Lexical Match: the final `ANSWER:` and the gold answer and synonyms are normalized
(case, punctuation, articles, whitespace) and must match exactly.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from spuma.config import PROJECT_ROOT, get_settings
from spuma.dataset.base import AdapterDataset, DatasetExample, Problem
from spuma.dataset.morehopqa.grading import normalize_answer
from spuma.dataset.qa import QAGrader

logger = logging.getLogger(__name__)

SPLIT_SOURCES = {"train": "dev", "validation": "dev", "test": "dev"}
CANDIDATES = {
    "id": ("id", "qid", "question_id", "ID", "QID", "Id"),
    "question": ("question", "Question", "query", "Query"),
    "answer": ("answer", "Answer", "answers", "Answers", "gold_answer", "short_answer"),
    "synonyms": ("synonyms", "Synonyms", "answer_synonyms", "aliases", "answer_aliases", "lexicon"),
}
OPTIONAL = ("answer", "synonyms")


def read_rows(path: str) -> list[dict[str, Any]]:
    p = Path(path) if Path(path).is_absolute() else PROJECT_ROOT / path
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        rows = data.get("data", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            raise ValueError(f"{p}: expected a JSON list of questions (or {{'data': [...]}})")
        return rows
    if p.suffix == ".jsonl":
        return list(load_dataset("json", data_files=str(p), split="train"))
    if p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        return list(load_dataset("csv", data_files=str(p), sep=sep, split="train"))
    raise ValueError(f"{p}: unsupported MedHopQA file type {p.suffix!r} (json, jsonl, csv, tsv)")


def detect_fields(columns: list[str], overrides: dict[str, str]) -> dict[str, str | None]:
    """{id, question, answer, synonyms: column or None}; fails on a missing or ambiguous required column."""
    out: dict[str, str | None] = {}
    for name, candidates in CANDIDATES.items():
        if name in overrides:
            if overrides[name] not in columns:
                raise ValueError(f"dataset.fields.{name} = {overrides[name]!r} is not a column; columns: {columns}")
            out[name] = overrides[name]
            continue
        found = [c for c in candidates if c in columns]
        if len(found) > 1:
            raise ValueError(f"ambiguous MedHopQA {name} column {found}; set dataset.fields.{name}; columns: {columns}")
        if not found and name not in OPTIONAL:
            raise ValueError(f"no MedHopQA {name} column; set dataset.fields.{name}; columns: {columns}")
        out[name] = found[0] if found else None
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return [s.strip() for s in re.split(r"\s*[|;]\s*", str(value)) if s.strip()]


class MedHopQAAdapter:
    def __init__(self, config: Any, loader: Any = None) -> None:
        self.config, self.loader = config, loader or read_rows

    def examples(self, official_split: str) -> list[DatasetExample]:
        c = self.config
        files = c.data_files if isinstance(c.data_files, dict) else {}
        if official_split not in files:
            raise ValueError(
                f"dataset.data_files names no file for the MedHopQA {official_split!r} split (download it from "
                "the BioCreative IX CodaBench competition)"
            )
        rows = self.loader(files[official_split])
        columns = sorted({k for r in rows for k in r})
        fields = detect_fields(columns, c.fields)
        logger.info("MedHopQA columns", extra={"split": official_split, "fields": fields, "columns": columns})
        out, unlabeled = [], 0
        for row in rows:
            answers = _as_list(row.get(fields["answer"])) if fields["answer"] else []
            synonyms = _as_list(row.get(fields["synonyms"])) if fields["synonyms"] else []
            if not answers:
                unlabeled += 1
                if c.labeled_only:
                    continue
            out.append(
                DatasetExample(
                    problem_id=f"medhopqa/{official_split}/{row[fields['id']]}",
                    dataset="medhopqa",
                    question=str(row[fields["question"]]),
                    choices=None,
                    context=None,
                    subject=None,
                    answer_type="open_ended",
                    reference_answer=answers[0] if answers else None,
                    grading_metadata={"source_split": official_split, "synonyms": answers[1:] + synonyms},
                )
            )
        logger.info(
            "MedHopQA split loaded",
            extra={"split": official_split, "kept": len(out), "unlabeled": unlabeled, "labeled_only": c.labeled_only},
        )
        return out


class MedHopQADataset(AdapterDataset):
    def __init__(self, loader: Any = None, config: Any = None) -> None:
        config = config or get_settings().dataset
        if config.train_samples or config.validation_samples:
            raise ValueError(
                "MedHopQA has no labeled training data: set dataset.train_samples and validation_samples to 0 and "
                "evaluate with the frozen MATH-trained models (chmm.frozen_from)"
            )
        super().__init__(MedHopQAAdapter(config, loader), SPLIT_SOURCES, config)


def _match(predicted: str, problem: Problem) -> tuple[bool, str]:
    golds = [problem.reference_solution, *problem.metadata["grading"]["synonyms"]]
    return normalize_answer(predicted) in {normalize_answer(g) for g in golds}, "lexical_match"


class MedHopQAGrader(QAGrader):
    def __init__(self) -> None:
        super().__init__(_match)
