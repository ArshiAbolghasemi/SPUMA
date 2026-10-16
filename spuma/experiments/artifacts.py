"""Artifact storage: JSON / JSONL files under one experiment directory, written atomically (safe to resume)."""

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol


class ArtifactStore(Protocol):
    root: Path

    def exists(self, rel: str) -> bool: ...

    def write_json(self, rel: str, obj: Any) -> Path: ...

    def read_json(self, rel: str) -> Any: ...

    def path(self, rel: str) -> Path: ...


class FileArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root.resolve()):
            raise ValueError(f"artifact path escapes the store: {rel!r}")
        return p

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def write_json(self, rel: str, obj: Any) -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=1, default=str)
        os.replace(tmp, p)
        return p

    def read_json(self, rel: str) -> Any:
        return json.loads(self.path(rel).read_text())

    def write_jsonl(self, rel: str, rows: Iterable[Any]) -> Path:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        os.replace(tmp, p)
        return p

    def append_jsonl(self, rel: str, row: Any) -> Path:
        """One line appended (progress logs; a crash loses at most the line being written)."""
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return p

    def read_jsonl(self, rel: str) -> list[Any]:
        return [json.loads(line) for line in self.path(rel).read_text().splitlines() if line]

    def remove(self, rel: str) -> None:
        self.path(rel).unlink(missing_ok=True)
