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

from spuma.config import get_settings
from spuma.models.config import BackendKind, GenerationConfig, ModelConfig

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
    content: str  # text with the reasoning removed
    reasoning: str


@runtime_checkable
class ModelBackend(Protocol):
    name: str  # backend kind
    info: BackendInfo
    capabilities: BackendCapabilities
    scorer_id: str

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        """Prompt token ids (chat template + generation prompt): the exact model input."""
        ...

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        """`request.n` continuations of one prompt."""
        ...

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[list[GenerationResult]]: ...

    def score(self, prompt_ids: list[int], token_ids: list[int], full_vocabulary: bool) -> TokenScores: ...

    def parse(self, text: str) -> ParsedOutput: ...

    def count_tokens(self, text: str) -> int: ...

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


def generation_logprob_mode(config: ModelConfig) -> LogprobMode:
    """What to score while generating `config`'s model: only when its generation backend is also its scoring
    backend, and only what the enabled `uncertainty:` metrics need."""
    uq = get_settings().uncertainty
    if not config.scores_at_generation:
        return "none"
    if uq.predictive_entropy.enabled:
        return "full"
    return "token" if uq.nll.enabled else "none"
