"""Hugging Face Transformers backend.

Generation samples with the configured warpers (temperature, top-p; top-k off unless set). Scores
always come from a separate teacher-forced forward pass over prompt + continuation, read at the
continuation positions from the *raw* model distribution (no temperature, no truncation):

    log p_t = log_softmax(logits_t),  NLL_t = -log p_t(y_t),  H_t = -Σ_v p_t(v) log p_t(v)

The vocabulary softmax is processed in chunks on the device; only per-token log-probs and entropies
are moved to the CPU.
"""

import gc
import logging
import time
from collections.abc import Sequence
from importlib import metadata
from typing import Any

import torch
from langchain_core.messages import BaseMessage
from transformers import AutoModelForCausalLM, AutoTokenizer

from spuma.common.config import to_plain
from spuma.models.backend import (
    BackendCapabilities,
    BackendInfo,
    GenerationRequest,
    GenerationResult,
    ParsedOutput,
    TokenScores,
    scorer_id,
)
from spuma.models.chat import ChatFormatter
from spuma.models.config import ModelConfig

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {requested!r} requested but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"device {requested!r} requested but MPS is not available")
    return device


class TransformersBackend:
    name = "transformers"

    @staticmethod
    def capabilities_for(config: ModelConfig) -> BackendCapabilities:
        return BackendCapabilities(
            generation=True,
            batch_generation=True,
            multiple_completions=True,
            generated_token_logprobs=True,
            full_vocabulary_logprobs=True,
            teacher_forced_scoring=True,
            full_vocabulary_scoring=True,
        )

    def __init__(self, model: Any, tokenizer: Any, config: ModelConfig) -> None:
        self.model = model.eval()
        self.config = config
        self.chat = ChatFormatter(tokenizer, config.enable_thinking)
        self.capabilities = self.capabilities_for(config)
        self.scorer_id = scorer_id(config, "transformers")
        eos = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(self.eos_ids)
        self.info = BackendInfo(
            backend=self.name,
            model=config.name,
            revision=config.revision,
            tokenizer=getattr(tokenizer, "name_or_path", config.name) or config.name,
            tokenizer_revision=config.tokenizer_revision or config.revision,
            chat_template_hash=self.chat.template_hash,
            vocab_size=self.chat.vocab_size,
            enable_thinking=config.enable_thinking,
            dtype=str(next(model.parameters()).dtype).removeprefix("torch."),
            library_version=metadata.version("transformers"),
        )

    @classmethod
    def from_config(cls, config: ModelConfig) -> TransformersBackend:

        tc = config.transformers
        device = resolve_device(tc.device)
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(config.name, revision=config.tokenizer_revision or config.revision)
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            revision=config.revision,
            dtype=getattr(torch, config.dtype),
            attn_implementation=tc.attn_implementation,
        ).to(device)
        logger.info(
            "transformers model loaded",
            extra={
                "model": config.name,
                "revision": config.revision,
                "device": str(device),
                "dtype": config.dtype,
                "seconds": round(time.perf_counter() - start, 2),
                **gpu_memory(),
            },
        )
        return cls(model, tokenizer, config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        return self.chat.render(messages)

    @torch.inference_mode()
    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        params, n = request.params, request.n
        if n > 1 and not params.do_sample:
            raise ValueError("n > 1 samples need temperature > 0")
        start = time.perf_counter()
        torch.manual_seed(request.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(request.seed)
        inputs = torch.tensor([request.prompt_ids], device=self.device)
        sampling = {"temperature": params.temperature, "top_p": params.top_p} if params.do_sample else {}
        out = self.model.generate(
            inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=params.do_sample,
            top_k=params.top_k or 0,
            max_new_tokens=params.max_new_tokens,
            num_return_sequences=n,
            pad_token_id=self.pad_id,
            eos_token_id=sorted(self.eos_ids),
            **sampling,
        )
        latency = (time.perf_counter() - start) / n
        results = []
        for row in out[:, len(request.prompt_ids) :].tolist():
            ids, finished = self._trim(row)
            lps = ents = None
            if request.logprobs != "none":
                scores = self.score(request.prompt_ids, ids, full_vocabulary=request.logprobs == "full")
                lps, ents = scores.token_logprobs, scores.token_entropies
            results.append(
                GenerationResult(
                    token_ids=ids,
                    text=self.chat.decode(ids),
                    finish_reason="stop" if finished else "length",
                    prompt_tokens=len(request.prompt_ids),
                    backend=self.name,
                    model=self.config.name,
                    tokenizer=self.info.tokenizer,
                    seed=request.seed,
                    params=to_plain(params),
                    logprob_mode=request.logprobs,
                    token_logprobs=lps,
                    token_entropies=ents,
                    scorer=self.scorer_id if lps is not None else None,
                    latency_s=latency,
                )
            )
        return results

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[list[GenerationResult]]:
        return [self.generate(r) for r in requests]

    def _trim(self, row: list[int]) -> tuple[list[int], bool]:
        for i, t in enumerate(row):
            if t in self.eos_ids:
                return row[: i + 1], True
            if t == self.pad_id:
                return row[:i], True
        return row, False

    @torch.inference_mode()
    def score(self, prompt_ids: list[int], token_ids: list[int], full_vocabulary: bool) -> TokenScores:
        start = time.perf_counter()
        if not token_ids:
            return TokenScores([], [] if full_vocabulary else None, self.scorer_id)
        ids = torch.tensor([prompt_ids + token_ids], device=self.device)
        logits = self.model(ids, logits_to_keep=len(token_ids) + 1).logits[0, :-1]
        target = torch.tensor(token_ids, device=self.device)
        size = self.config.transformers.scoring_chunk_size
        lps, ents = [], []
        for s in range(0, len(token_ids), size):
            chunk = torch.log_softmax(logits[s : s + size].float(), dim=-1)
            lps.append(chunk.gather(1, target[s : s + chunk.shape[0], None]).squeeze(1))
            if full_vocabulary:
                probs = chunk.exp()
                ents.append(-torch.where(probs > 0, probs * chunk, 0.0).sum(-1))
        return TokenScores(
            token_logprobs=torch.cat(lps).tolist(),
            token_entropies=torch.cat(ents).tolist() if full_vocabulary else None,
            scorer=self.scorer_id,
            latency_s=time.perf_counter() - start,
        )

    def parse(self, text: str) -> ParsedOutput:
        return self.chat.parse(text)

    def count_tokens(self, text: str) -> int:
        return self.chat.count_tokens(text)

    def encode(self, text: str) -> list[int]:
        return self.chat.encode(text)

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def gpu_memory() -> dict[str, float]:
    """Current/peak CUDA memory in GiB (empty when no CUDA device)."""
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    return {
        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / gib, 3),
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
    }
