"""`model:` / `models:` sections: model identity, generation settings and the two inference backends."""

from dataclasses import dataclass, field
from typing import Literal

from spuma.common.config import check_choice, check_range, to_plain
from spuma.common.utils import stable_hash

BackendKind = Literal["transformers", "vllm"]
BACKENDS = ("transformers", "vllm")


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling settings of a model call (`model.generation:`, or an agent's own `generation:`)."""

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None  # None/0 = disabled (no top-k truncation)
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        check_range("generation.temperature", self.temperature, ge=0.0)
        check_range("generation.top_p", self.top_p, gt=0.0, le=1.0)
        check_range("generation.top_k", self.top_k, ge=0)
        check_range("generation.max_new_tokens", self.max_new_tokens, gt=0)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


@dataclass(frozen=True)
class TransformersBackendConfig:
    """`model.transformers:` section."""

    device: str = "auto"
    attn_implementation: str = "sdpa"
    # Rows of the full-vocabulary softmax processed at once during scoring (GPU memory bound).
    scoring_chunk_size: int = 256

    def __post_init__(self) -> None:
        check_choice(
            "model.transformers.attn_implementation", self.attn_implementation, ("sdpa", "eager", "flash_attention_2")
        )
        check_range("model.transformers.scoring_chunk_size", self.scoring_chunk_size, gt=0)


@dataclass(frozen=True)
class VLLMBackendConfig:
    """`model.vllm:` section."""

    max_model_len: int | None = 8192
    gpu_memory_utilization: float = 0.85
    tensor_parallel_size: int = 1
    # -1 = whole vocabulary; required for full-vocabulary PE from vLLM (see README for the cost).
    max_logprobs: int = -1
    enforce_eager: bool = False

    def __post_init__(self) -> None:
        check_range("model.vllm.max_model_len", self.max_model_len, gt=0)
        check_range("model.vllm.gpu_memory_utilization", self.gpu_memory_utilization, gt=0.0, le=1.0)
        check_range("model.vllm.tensor_parallel_size", self.tensor_parallel_size, gt=0)
        check_range("model.vllm.max_logprobs", self.max_logprobs, ge=-1)


@dataclass(frozen=True)
class ModelConfig:
    """`model:` section (and each entry of `models:`)."""

    name: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    tokenizer_revision: str | None = None  # None: same as `revision`
    backend: str = "transformers"  # generation (agents + semantic-entropy sampling)
    # PE/NLL scoring: `same` scores at generation time with `backend`; a different kind re-scores the
    # saved prompt/continuation token ids teacher-forced in a later, separately loaded stage.
    scoring_backend: str = "same"
    dtype: str = "bfloat16"
    enable_thinking: bool = False
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    transformers: TransformersBackendConfig = field(default_factory=TransformersBackendConfig)
    vllm: VLLMBackendConfig = field(default_factory=VLLMBackendConfig)

    def __post_init__(self) -> None:
        check_choice("model.backend", self.backend, BACKENDS)
        check_choice("model.scoring_backend", self.scoring_backend, ("same", *BACKENDS))
        check_choice("model.dtype", self.dtype, ("bfloat16", "float16", "float32"))

    @property
    def scoring_kind(self) -> BackendKind:
        return self.backend if self.scoring_backend == "same" else self.scoring_backend  # type: ignore[return-value]

    @property
    def scores_at_generation(self) -> bool:
        return self.scoring_kind == self.backend

    def fingerprint(self, kind: BackendKind | None = None) -> str:
        """Identity of a loaded engine: shared fields + that backend's section (generation settings excluded)."""
        kind = kind or self.backend  # type: ignore[assignment]
        data = to_plain(self)
        shared = {k: data[k] for k in ("name", "revision", "tokenizer_revision", "dtype", "enable_thinking")}
        return stable_hash({**shared, "kind": kind, kind: data[kind]})

    def generation_fingerprint(self) -> str:
        """Everything that can change generated trajectories (and generation-time scores)."""
        unused = "vllm" if self.backend == "transformers" else "transformers"
        data = {k: v for k, v in to_plain(self).items() if k not in ("scoring_backend", unused)}
        data["scores_at_generation"] = self.scores_at_generation
        return stable_hash(data)
