"""vLLM backend (offline `LLM` engine; checked against vLLM 0.31).

What is used and why (from the vLLM 0.31 source):
  - `logprobs_mode="raw_logprobs"` (engine): sample log-probs are taken *before* temperature, top-p
    and other processors, i.e. the raw model distribution p_theta(.|c_t) that PE/NLL are defined on.
    Prompt log-probs never pass through processors, so teacher-forced scores are raw as well.
  - `logprobs=0`  -> the sampled token's raw log-prob (NLL);
    `logprobs=-1` -> the whole vocabulary at every generated position (PE). Needs engine `max_logprobs=-1`.
  - `prompt_logprobs` on prompt+continuation gives teacher-forced scores of a fixed continuation.
  - `top_k=0` disables top-k (vLLM 0.31 rejects -1); `detokenize=False` skips vLLM's per-logprob token
    decoding (text is decoded with the same HF tokenizer as the Transformers backend);
    `flat_logprobs=True` returns flat lists instead of one dict per position.

Cost of full-vocabulary log-probs: vLLM has no supported way to reduce the distribution to an entropy
on the GPU, so PE moves |V| (~152k for Qwen3) floats per scored position to the CPU:
  - at generation (`scoring_backend: same`): ~|V| x generated tokens per step;
  - teacher-forced (`scoring_backend: vllm` with another generator): |V| x (prompt + continuation) tokens,
    because prompt log-probs cover every prompt position. This is usually prohibitive for long prompts;
    prefer `scoring_backend: transformers` (hybrid mode) for PE.
Distributions are never truncated: a position that does not carry the whole vocabulary raises.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterator, Sequence
from importlib import metadata
from typing import Any

import numpy as np
import torch
from langchain_core.messages import BaseMessage
from transformers import AutoTokenizer

from spuma.common.config import to_plain
from spuma.models.backend import (
    BackendCapabilities,
    BackendCapabilityError,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    LogprobMode,
    ParsedOutput,
    TokenScores,
    scorer_id,
)
from spuma.models.chat import ChatFormatter
from spuma.models.config import GenerationConfig, ModelConfig

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
except ImportError:
    LLM = SamplingParams = TokensPrompt = None
try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except ImportError:
    destroy_distributed_environment = destroy_model_parallel = None

logger = logging.getLogger(__name__)

REQUIRED_SAMPLING_FIELDS = ("n", "seed", "top_k", "logprobs", "prompt_logprobs", "flat_logprobs", "detokenize")
INSTALL_HINT = "install it with `scripts/install-dependencies.sh --vllm` (Linux + NVIDIA GPU)"


def _sampling_fields(cls: Any) -> set[str]:
    return set(getattr(cls, "__struct_fields__", ()) or getattr(cls, "__dataclass_fields__", {}))


class VLLMBackend:
    name = "vllm"

    @staticmethod
    def capabilities_for(config: ModelConfig) -> BackendCapabilities:
        if SamplingParams is None:
            raise BackendCapabilityError(f"vLLM is not installed: {INSTALL_HINT}")
        missing = [f for f in REQUIRED_SAMPLING_FIELDS if f not in _sampling_fields(SamplingParams)]
        if missing:
            raise BackendCapabilityError(f"installed vLLM {_version()} lacks SamplingParams fields {missing}")
        try:
            SamplingParams(top_k=0, logprobs=-1, flat_logprobs=True, detokenize=False, seed=0)
        except Exception as e:  # noqa: BLE001
            raise BackendCapabilityError(f"installed vLLM {_version()} rejects required sampling params: {e}") from e
        full = config.vllm.max_logprobs == -1
        return BackendCapabilities(
            generation=True,
            batch_generation=True,
            multiple_completions=True,
            generated_token_logprobs=True,
            full_vocabulary_logprobs=full,
            teacher_forced_scoring=True,
            full_vocabulary_scoring=full,
        )

    def __init__(
        self,
        llm: Any,
        tokenizer: Any,
        config: ModelConfig,
        sampling_params_cls: Any,
        tokens_prompt_cls: Any,
        engine_vocab_size: int | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.chat = ChatFormatter(tokenizer, config.enable_thinking)
        self.SamplingParams = sampling_params_cls
        self.TokensPrompt = tokens_prompt_cls
        self.capabilities = self.capabilities_for(config)
        self.scorer_id = scorer_id(config, "vllm")
        self.engine_vocab_size = engine_vocab_size or llm.llm_engine.model_config.get_vocab_size()
        self.info = BackendInfo(
            backend=self.name,
            model=config.name,
            revision=config.revision,
            tokenizer=getattr(tokenizer, "name_or_path", config.name) or config.name,
            tokenizer_revision=config.tokenizer_revision or config.revision,
            chat_template_hash=self.chat.template_hash,
            vocab_size=self.chat.vocab_size,
            enable_thinking=config.enable_thinking,
            dtype=config.dtype,
            library_version=_version(),
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> VLLMBackend:
        cls.capabilities_for(config)
        vc = config.vllm
        tokenizer_revision = config.tokenizer_revision or config.revision
        start = time.perf_counter()
        llm = LLM(
            model=config.name,
            revision=config.revision,
            tokenizer_revision=tokenizer_revision,
            dtype=config.dtype,
            max_model_len=vc.max_model_len,
            gpu_memory_utilization=vc.gpu_memory_utilization,
            tensor_parallel_size=vc.tensor_parallel_size,
            max_logprobs=vc.max_logprobs,
            logprobs_mode="raw_logprobs",
            enforce_eager=vc.enforce_eager,
        )
        tokenizer = AutoTokenizer.from_pretrained(config.name, revision=tokenizer_revision)
        logger.info(
            "vllm engine loaded",
            extra={
                "model": config.name,
                "revision": config.revision,
                "vllm": _version(),
                "max_model_len": vc.max_model_len,
                "max_logprobs": vc.max_logprobs,
                "seconds": round(time.perf_counter() - start, 2),
            },
        )
        return cls(llm, tokenizer, config, SamplingParams, TokensPrompt)

    def sampling_params(self, params: GenerationConfig, n: int, seed: int, logprobs: LogprobMode) -> Any:
        kw: dict[str, Any] = {
            "n": n,
            "max_tokens": params.max_new_tokens,
            "seed": seed,
            "detokenize": False,
            "flat_logprobs": True,
        }
        if params.do_sample:
            kw |= {"temperature": params.temperature, "top_p": params.top_p, "top_k": params.top_k or 0}
        else:
            if n > 1:
                raise ValueError("n > 1 samples need temperature > 0")
            kw |= {"temperature": 0.0}
        if logprobs == "full":
            self._require_full("full-vocabulary generation log-probs")
            kw["logprobs"] = -1
        elif logprobs == "token":
            kw["logprobs"] = 0
        return self.SamplingParams(**kw)

    def _require_full(self, what: str) -> None:
        if self.config.vllm.max_logprobs != -1:
            raise BackendCapabilityError(
                f"{what} need model.vllm.max_logprobs: -1 (got {self.config.vllm.max_logprobs})"
            )

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        return self.chat.render(messages)

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[list[GenerationResult]]:
        if not requests:
            return []
        start = time.perf_counter()
        prompts = [self.TokensPrompt(prompt_token_ids=list(r.prompt_ids)) for r in requests]
        params = [self.sampling_params(r.params, r.n, r.seed, r.logprobs) for r in requests]
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        latency = (time.perf_counter() - start) / sum(r.n for r in requests)
        results = []
        for request, output in zip(requests, outputs, strict=True):
            if len(output.outputs) != request.n:
                raise BackendCapabilityError(f"vLLM returned {len(output.outputs)} completions, expected {request.n}")
            results.append([self._result(request, c, latency) for c in output.outputs])
        return results

    def _result(self, request: GenerationRequest, completion: Any, latency: float) -> GenerationResult:
        ids = list(completion.token_ids)
        lps = ents = None
        if request.logprobs != "none":
            if completion.logprobs is None or len(completion.logprobs) != len(ids):
                raise BackendCapabilityError("vLLM returned no or misaligned log-probs for the generated tokens")
            lps, ents = self._extract(_positions(completion.logprobs), ids, request.logprobs == "full")
        return GenerationResult(
            token_ids=ids,
            text=self.chat.decode(ids),
            finish_reason=completion.finish_reason or "stop",
            prompt_tokens=len(request.prompt_ids),
            backend=self.name,
            model=self.config.name,
            tokenizer=self.info.tokenizer,
            seed=request.seed,
            params=to_plain(request.params),
            logprob_mode=request.logprobs,
            token_logprobs=lps,
            token_entropies=ents,
            scorer=self.scorer_id if lps is not None else None,
            latency_s=latency,
        )

    def score(self, prompt_ids: list[int], token_ids: list[int], full_vocabulary: bool) -> TokenScores:
        start = time.perf_counter()
        if not token_ids:
            return TokenScores([], [] if full_vocabulary else None, self.scorer_id)
        total = len(prompt_ids) + len(token_ids)
        limit = self.config.vllm.max_model_len
        if limit is not None and total + 1 > limit:
            raise BackendCapabilityError(f"scoring needs {total + 1} positions > model.vllm.max_model_len={limit}")
        if full_vocabulary:
            self._require_full("full-vocabulary teacher-forced scoring")
            logger.warning(
                "vLLM full-vocabulary prompt log-probs transfer every prompt position",
                extra={"positions": total, "floats": total * self.engine_vocab_size},
            )
        params = self.SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=-1 if full_vocabulary else 1,
            flat_logprobs=True,
            detokenize=False,
        )
        (output,) = self.llm.generate(
            [self.TokensPrompt(prompt_token_ids=prompt_ids + token_ids)], params, use_tqdm=False
        )
        positions = list(_positions(output.prompt_logprobs))
        if len(positions) != total:
            raise BackendCapabilityError(f"vLLM returned {len(positions)} prompt log-prob rows for {total} tokens")
        lps, ents = self._extract(iter(positions[len(prompt_ids) :]), token_ids, full_vocabulary)
        return TokenScores(lps, ents, self.scorer_id, time.perf_counter() - start)

    def _extract(
        self, positions: Iterator[tuple[np.ndarray, np.ndarray]], token_ids: list[int], full: bool
    ) -> tuple[list[float], list[float] | None]:
        lps, ents = [], []
        for t, (ids, vals) in zip(range(len(token_ids)), positions, strict=False):
            target = token_ids[t]
            hit = np.flatnonzero(ids == target)
            if not len(hit):
                raise BackendCapabilityError(f"log-prob of token {target} missing at continuation position {t}")
            lps.append(float(vals[hit[0]]))
            if full:
                ents.append(self._entropy(ids, vals, t))
        if len(lps) != len(token_ids):
            raise BackendCapabilityError(f"got log-probs for {len(lps)} of {len(token_ids)} tokens")
        return lps, (ents if full else None)

    def _entropy(self, ids: np.ndarray, vals: np.ndarray, position: int) -> float:
        _, first = np.unique(ids, return_index=True)
        if len(first) != self.engine_vocab_size:
            raise BackendCapabilityError(
                f"position {position}: got {len(first)} distinct log-probs for a vocabulary of "
                f"{self.engine_vocab_size}; refusing to compute a truncated entropy"
            )
        lp = vals[first]
        p = np.exp(lp)
        return float(-np.sum(np.where(p > 0, p * lp, 0.0)))

    def parse(self, text: str) -> ParsedOutput:
        return self.chat.parse(text)

    def count_tokens(self, text: str) -> int:
        return self.chat.count_tokens(text)

    def encode(self, text: str) -> list[int]:
        return self.chat.encode(text)

    def close(self) -> None:
        llm, self.llm = self.llm, None
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        del llm
        if destroy_model_parallel is not None:
            destroy_model_parallel()
            destroy_distributed_environment()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _positions(container: Any) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(token ids, log-probs) per position from vLLM FlatLogprobs or a list of {token_id: Logprob}."""
    if hasattr(container, "start_indices"):
        ids = np.asarray(container.token_ids, dtype=np.int64)
        vals = np.asarray(container.logprobs, dtype=np.float64)
        for s, e in zip(container.start_indices, container.end_indices, strict=True):
            yield ids[s:e], vals[s:e]
        return
    for pos in container:
        if not pos:
            yield np.empty(0, np.int64), np.empty(0)
        else:
            yield np.fromiter(pos.keys(), np.int64), np.fromiter((v.logprob for v in pos.values()), np.float64)


def _version() -> str | None:
    try:
        return metadata.version("vllm")
    except metadata.PackageNotFoundError:
        return None
