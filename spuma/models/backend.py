"""Model-backend interface, typed request/result objects, capabilities, and the engine cache.

Agents and uncertainty code only see these types; `transformers` / `vllm` imports stay inside the
backend implementations. One engine is created per (backend kind, model fingerprint) and shared by
every agent that uses it.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage

from spuma.models.config import BackendKind, GenerationConfig, ModelConfig
from spuma.uncertainty.config import UncertaintyConfig

logger = logging.getLogger(__name__)

LogprobMode = Literal["none", "token", "full"]  # what to return with each generated token


class BackendCapabilityError(RuntimeError):
    """A required capability is unavailable: raised instead of returning approximate scores."""


@dataclass(frozen=True)
class BackendCapabilities:
    generation: bool = True
    batch_generation: bool = False
    multiple_completions: bool = False
    generated_token_logprobs: bool = False  # raw log p of each generated token, at generation time
    full_vocabulary_logprobs: bool = False  # raw full distribution at each generated position
    teacher_forced_scoring: bool = False  # log p of a given continuation after a given prompt
    full_vocabulary_scoring: bool = False  # ... plus the full distribution at each scored position
    tool_calling: bool = False  # tool schemas in the prompt + Qwen3 tool-call parsing

    def missing(self, *names: str) -> list[str]:
        known = {f.name for f in fields(self)}
        unknown = set(names) - known
        if unknown:
            raise ValueError(f"unknown capabilities {sorted(unknown)}")
        return [n for n in names if not getattr(self, n)]


@dataclass(frozen=True)
class BackendInfo:
    backend: str
    model: str
    revision: str | None
    tokenizer: str
    tokenizer_revision: str | None
    chat_template_hash: str
    vocab_size: int
    enable_thinking: bool
    dtype: str
    library_version: str | None

    def compatibility_key(self) -> dict[str, Any]:
        """Fields that must match for two backends to produce/score the same token sequences."""
        d = asdict(self)
        return {
            k: d[k]
            for k in ("model", "revision", "tokenizer", "tokenizer_revision", "chat_template_hash", "enable_thinking")
        }


@dataclass(frozen=True)
class GenerationRequest:
    prompt_ids: list[int]
    params: GenerationConfig
    seed: int
    n: int = 1
    logprobs: LogprobMode = "none"


@dataclass(frozen=True)
class GenerationResult:
    """One continuation, normalized across backends. Log-probs/entropies are of the raw model distribution."""

    token_ids: list[int]  # model-generated tokens only (EOS kept when generated)
    text: str
    finish_reason: str  # stop | length
    prompt_tokens: int
    backend: str
    model: str
    tokenizer: str
    seed: int
    params: dict[str, Any]
    logprob_mode: LogprobMode = "none"
    token_logprobs: list[float] | None = None
    token_entropies: list[float] | None = None
    scorer: str | None = None  # scorer id when log-probs are present
    latency_s: float = 0.0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def metadata(self) -> dict[str, Any]:
        keys = (
            "finish_reason",
            "prompt_tokens",
            "backend",
            "model",
            "tokenizer",
            "logprob_mode",
            "scorer",
            "latency_s",
        )
        return {k: getattr(self, k) for k in keys} | {"num_tokens": self.num_tokens}


@dataclass(frozen=True)
class TokenScores:
    """Teacher-forced raw-model scores of a fixed continuation after a fixed prompt."""

    token_logprobs: list[float]
    token_entropies: list[float] | None  # None unless full-vocabulary scoring was requested
    scorer: str
    latency_s: float = 0.0


@dataclass(frozen=True)
class ParsedOutput:
    content: str  # text with reasoning and tool-call markup removed
    reasoning: str
    tool_calls: list[dict[str, Any]]  # {"name", "args", "id"}
    invalid_tool_calls: list[dict[str, Any]]  # {"name", "args" (raw str), "id", "error"}


@runtime_checkable
class ModelBackend(Protocol):
    name: str  # backend kind
    info: BackendInfo
    capabilities: BackendCapabilities
    scorer_id: str

    def render(self, messages: Sequence[BaseMessage], tools: list[dict[str, Any]] | None) -> list[int]:
        """Prompt token ids (chat template + tool schemas + generation prompt): the exact model input."""
        ...

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        """`request.n` continuations of one prompt."""
        ...

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[list[GenerationResult]]: ...

    def score(self, prompt_ids: list[int], token_ids: list[int], full_vocabulary: bool) -> TokenScores: ...

    def parse(self, text: str) -> ParsedOutput: ...

    def count_tokens(self, text: str) -> int: ...

    def truncate(self, text: str, max_tokens: int) -> str: ...

    def close(self) -> None: ...


class BackendClass(Protocol):
    @staticmethod
    def capabilities_for(config: ModelConfig) -> BackendCapabilities:
        """Capabilities for this config and the installed library, without loading the model."""
        ...

    @classmethod
    def from_config(cls, config: ModelConfig) -> ModelBackend: ...


def require(backend: ModelBackend, *capabilities: str) -> None:
    missing = backend.capabilities.missing(*capabilities)
    if missing:
        raise BackendCapabilityError(f"backend {backend.name!r} lacks required capabilities {missing}")


def scorer_id(config: ModelConfig, kind: BackendKind) -> str:
    """Identifier stored with every score so cached scores are reused only by the same scoring engine."""
    return f"{kind}:{config.fingerprint(kind)}"


def generation_logprob_mode(config: ModelConfig, uq: UncertaintyConfig) -> LogprobMode:
    """What to score while generating: only when the generation backend is also the scoring backend."""
    if not config.scores_at_generation:
        return "none"
    if uq.predictive_entropy.enabled:
        return "full"
    return "token" if uq.nll.enabled else "none"


_CLASSES: dict[str, BackendClass] = {}
_INSTANCES: dict[tuple[str, str], ModelBackend] = {}


def register_backend(name: str, cls: BackendClass) -> None:
    _CLASSES[name] = cls


def backend_class(kind: str) -> BackendClass:
    if kind not in _CLASSES:
        if kind == "transformers":
            from spuma.models import transformers_backend  # noqa: F401  (registers itself)
        elif kind == "vllm":
            from spuma.models import vllm_backend  # noqa: F401
        else:
            raise BackendCapabilityError(f"no backend registered under {kind!r}")
    return _CLASSES[kind]


def get_backend(config: ModelConfig, kind: BackendKind | None = None) -> ModelBackend:
    """Create an engine once per (kind, model fingerprint); later calls reuse it."""
    kind = kind or config.backend
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
