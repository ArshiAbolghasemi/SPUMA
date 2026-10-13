"""Engine registry and cache: backend kind -> class, and one loaded engine per (kind, model fingerprint).

Agents sharing a model share its engine. `release_backends()` frees GPU memory between the generate and
score phases of the hybrid mode (vLLM generation, Transformers scoring).
"""

import logging

from spuma.models.backend import BackendCapabilityError, BackendClass, ModelBackend
from spuma.models.config import BackendKind, ModelConfig
from spuma.models.transformers_backend import TransformersBackend
from spuma.models.vllm_backend import VLLMBackend

logger = logging.getLogger(__name__)

BACKENDS: dict[str, BackendClass] = {"transformers": TransformersBackend, "vllm": VLLMBackend}
_INSTANCES: dict[tuple[str, str], ModelBackend] = {}


def backend_class(kind: str) -> BackendClass:
    if kind not in BACKENDS:
        raise BackendCapabilityError(f"no backend registered under {kind!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[kind]


def get_backend(config: ModelConfig, kind: BackendKind | None = None) -> ModelBackend:
    """Create an engine once per (kind, model fingerprint); later calls reuse it."""
    kind = kind or config.backend  # type: ignore[assignment]
    key = (kind, config.fingerprint(kind))
    if key not in _INSTANCES:
        _INSTANCES[key] = backend_class(kind).from_config(config)
        logger.info("backend created", extra={"backend": kind, "model": config.name, "fingerprint": key[1]})
    return _INSTANCES[key]


def release_backends(kind: str | None = None) -> None:
    """Close and forget cached engines (all, or one kind) so another engine can use the GPU memory."""
    for key in [k for k in _INSTANCES if kind is None or k[0] == kind]:
        _INSTANCES.pop(key).close()
        logger.info("backend released", extra={"backend": key[0], "fingerprint": key[1]})


def clear_backends() -> None:
    _INSTANCES.clear()
