"""A stand-in `vllm` module for offline tests, backed by a real (tiny) HF causal LM.

It mimics the vLLM 0.31 offline API surface SPUMA uses: `LLM.generate(TokensPrompt, SamplingParams)`
returning outputs whose log-probs are FlatLogprobs of the *raw* model distribution, with the
sampled/prompt token listed first and then repeated inside the top-k / full-vocabulary entries.
"""

import importlib.machinery
import types
from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int | None = 16
    seed: int | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    flat_logprobs: bool = False
    detokenize: bool = True

    def __post_init__(self):
        if self.top_k != 0 and self.top_k < 1:  # vLLM 0.31 semantics
            raise ValueError(f"top_k must be 0 (disable), or at least 1, got {self.top_k}.")


@dataclass
class FlatLogprobs:
    start_indices: list = field(default_factory=list)
    end_indices: list = field(default_factory=list)
    token_ids: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)

    def append(self, chosen, row, k):
        """Chosen token first, then the top-k (k=-1: whole vocabulary) of the raw log-prob row."""
        self.start_indices.append(len(self.logprobs))
        if row is not None:
            ids = [chosen] + (
                torch.arange(row.numel()).tolist() if k == -1 else row.topk(k).indices.tolist() if k else []
            )
            self.token_ids += ids
            self.logprobs += [float(row[i]) for i in ids]
        self.end_indices.append(len(self.logprobs))

    def __len__(self):
        return len(self.start_indices)


@dataclass
class TokensPrompt:
    prompt_token_ids: list


@dataclass
class Completion:
    token_ids: list
    logprobs: FlatLogprobs | None
    finish_reason: str


@dataclass
class RequestOutput:
    prompt_token_ids: list
    prompt_logprobs: FlatLogprobs | None
    outputs: list


class LLM:
    def __init__(self, model, eos_ids, max_logprobs=-1):
        self.model, self.eos_ids, self.max_logprobs = model, set(eos_ids), max_logprobs
        vocab = model.config.vocab_size
        self.llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(get_vocab_size=lambda: vocab))
        self.requests = []

    @torch.no_grad()
    def _raw(self, ids):
        return torch.log_softmax(self.model(torch.tensor([ids])).logits[0].float(), -1)

    def generate(self, prompts, params, use_tqdm=False):
        out = []
        if not isinstance(params, list):  # vLLM accepts one SamplingParams for all prompts
            params = [params] * len(prompts)
        for prompt, sp in zip(prompts, params, strict=True):
            self.requests.append((prompt, sp))
            ids = prompt.prompt_token_ids
            prompt_lps = None
            if sp.prompt_logprobs is not None:
                raw = self._raw(ids)
                prompt_lps = FlatLogprobs()
                prompt_lps.append(None, None, 0)
                for i in range(1, len(ids)):
                    prompt_lps.append(ids[i], raw[i - 1], sp.prompt_logprobs)
            gen = torch.Generator().manual_seed(sp.seed or 0)
            completions = []
            for _ in range(sp.n):
                seq, flat = list(ids), FlatLogprobs() if sp.logprobs is not None else None
                for _ in range(sp.max_tokens):
                    raw = self._raw(seq)[-1]
                    if sp.temperature == 0:
                        tok = int(raw.argmax())
                    else:
                        tok = int(torch.multinomial(torch.softmax(raw / sp.temperature, -1), 1, generator=gen))
                    if flat is not None:
                        flat.append(tok, raw, sp.logprobs)
                    seq.append(tok)
                    if tok in self.eos_ids:
                        break
                gen_ids = seq[len(ids) :]
                reason = "stop" if gen_ids and gen_ids[-1] in self.eos_ids else "length"
                completions.append(Completion(gen_ids, flat, reason))
            out.append(RequestOutput(ids, prompt_lps, completions))
        return out


def module():
    m = types.ModuleType("vllm")
    m.__spec__ = importlib.machinery.ModuleSpec("vllm", None)
    m.SamplingParams, m.LLM = SamplingParams, LLM
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = TokensPrompt
    m.inputs = inputs
    return m, inputs
