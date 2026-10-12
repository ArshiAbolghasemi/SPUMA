"""`model:` / `models:` sections: model identity, generation settings and the two inference backends."""

from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig
from spuma.common.utils import stable_hash

BackendKind = Literal["transformers", "vllm"]


class GenerationConfig(StrictConfig):
    temperature: float = Field(0.7, ge=0.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    top_k: int | None = None  # None/0 = disabled (no top-k truncation)
    max_new_tokens: int = Field(2048, gt=0)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0


class TransformersBackendConfig(StrictConfig):
    device: str = "auto"
    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    # Rows of the full-vocabulary softmax processed at once during scoring (GPU memory bound).
    scoring_chunk_size: int = Field(256, gt=0)


class VLLMBackendConfig(StrictConfig):
    max_model_len: int | None = Field(8192, gt=0)
    gpu_memory_utilization: float = Field(0.85, gt=0.0, le=1.0)
    tensor_parallel_size: int = Field(1, gt=0)
    # -1 = whole vocabulary; required for full-vocabulary PE from vLLM (see README for the cost).
    max_logprobs: int = Field(-1, ge=-1)
    enforce_eager: bool = False


class ModelConfig(StrictConfig):
    """`model:` section (and each entry of `models:`)."""

    name: str = "Qwen/Qwen3-8B"
    revision: str | None = None
    tokenizer_revision: str | None = None  # None: same as `revision`
    backend: BackendKind = "transformers"  # generation (agents + semantic-entropy sampling)
    # PE/NLL scoring: `same` scores at generation time with `backend`; a different kind re-scores the
    # saved prompt/continuation token ids teacher-forced in a later, separately loaded stage.
    scoring_backend: Literal["same", "transformers", "vllm"] = "same"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    enable_thinking: bool = False
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    transformers: TransformersBackendConfig = Field(default_factory=TransformersBackendConfig)
    vllm: VLLMBackendConfig = Field(default_factory=VLLMBackendConfig)

    @property
    def scoring_kind(self) -> BackendKind:
        return self.backend if self.scoring_backend == "same" else self.scoring_backend

    @property
    def scores_at_generation(self) -> bool:
        return self.scoring_kind == self.backend

    def fingerprint(self, kind: BackendKind | None = None) -> str:
        """Identity of a loaded engine: shared fields + that backend's section (generation settings excluded)."""
        kind = kind or self.backend
        data = self.model_dump(
            mode="json", include={"name", "revision", "tokenizer_revision", "dtype", "enable_thinking"}
        )
        return stable_hash({**data, "kind": kind, kind: self.model_dump(mode="json")[kind]})

    def generation_fingerprint(self) -> str:
        """Everything that can change generated trajectories (and generation-time scores)."""
        exclude = {"scoring_backend", "vllm" if self.backend == "transformers" else "transformers"}
        data = self.model_dump(mode="json", exclude=exclude)
        data["scores_at_generation"] = self.scores_at_generation
        return stable_hash(data)
