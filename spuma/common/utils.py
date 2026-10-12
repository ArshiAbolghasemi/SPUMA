"""Small dependency-free helpers shared across packages."""

import hashlib
import json
from typing import Any


def stable_hash(data: Any, length: int = 12) -> str:
    """Short SHA-256 of the canonical JSON of `data` (used for config fingerprints)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:length]


def derive_seed(base: int, *parts: object) -> int:
    """Deterministic per-call seed from the run seed and call coordinates (problem, agent, step, ...)."""
    key = ":".join(str(p) for p in (base, *parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def safe_name(identifier: str) -> str:
    """Filesystem-safe file name for an id such as `math/test/algebra/3`."""
    return identifier.replace("/", "__")
