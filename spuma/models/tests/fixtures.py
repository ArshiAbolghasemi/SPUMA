"""Offline model fixtures for every test (registered in pyproject: `-p spuma.models.tests.fixtures`).

A byte-level BPE tokenizer trained in-process, a tiny randomly initialised Qwen3 causal LM (no downloads,
CPU), and a scripted backend that replays fixed outputs for workflow tests.
"""

import math
from collections.abc import Sequence

import pytest
import torch
from langchain_core.messages import BaseMessage
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

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
from spuma.models.chat import parse_qwen_output, to_chat_dicts
from spuma.models.config import ModelConfig, TransformersBackendConfig
from spuma.models.transformers_backend import TransformersBackend

CORPUS = [
    "Solve for x: 2x + 3 = 7. The answer is \\boxed{2}.",
    "Let x**2 - 4 = 0, so x = 2 or x = -2.",
    "Review the solution, verify calculations and give the final answer.",
    "<|im_start|> <|im_end|> system user assistant <think> </think>",
] * 20

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


TINY_CONFIG = ModelConfig(name="tiny-qwen3", dtype="float32", transformers=TransformersBackendConfig(device="cpu"))


@pytest.fixture(scope="session")
def tokenizer():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=["<|im_end|>", "<|im_start|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>")
    fast.chat_template = CHAT_TEMPLATE
    return fast


@pytest.fixture(scope="session")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval()


@pytest.fixture
def tiny_backend(tiny_model, tokenizer):
    return TransformersBackend(tiny_model, tokenizer, TINY_CONFIG)


ALL_CAPABILITIES = BackendCapabilities(
    generation=True,
    batch_generation=True,
    multiple_completions=True,
    generated_token_logprobs=True,
    full_vocabulary_logprobs=True,
    teacher_forced_scoring=True,
    full_vocabulary_scoring=True,
)


class ScriptedBackend:
    """Replays scripted generations in order; per-token stats are fixed so metrics are known exactly.

    `samples` maps a prompt substring to the texts returned for n-sample (semantic-entropy) calls.
    It identifies as the `kind` engine of `model` (default: the project config's model), so steps it
    scores at generation time count as scored by that engine.
    """

    capabilities = ALL_CAPABILITIES

    def __init__(
        self,
        outputs: list[str],
        samples: dict[str, list[str]] | None = None,
        logprob: float = -0.5,
        model: ModelConfig | None = None,
        kind: str = "transformers",
    ):
        self.outputs = list(outputs)
        self.samples = samples or {}
        self.logprob = logprob
        self.name = kind
        model = model or ModelConfig()
        self.scorer_id = scorer_id(model, kind)
        self.info = BackendInfo(kind, model.name, None, model.name, None, "tmpl", 128, False, "float32", None)
        self.prompts: list[str] = []
        self.calls: list[GenerationRequest] = []
        self.score_calls: list[tuple[list[int], list[int], bool]] = []
        self.closed = False

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_chat_dicts(messages))
        self.prompts.append(text)
        return [ord(c) for c in text]

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        self.calls.append(request)
        prompt = "".join(chr(i) for i in request.prompt_ids)
        if request.n == 1:
            texts = [self.outputs.pop(0)]
        else:
            key = next((k for k in self.samples if k in prompt), None)
            texts = self.samples[key][: request.n] if key is not None else [f"sample {i}" for i in range(request.n)]
        out = []
        for t in texts:
            ids = [ord(c) for c in t] or [0]
            lps = [self.logprob] * len(ids) if request.logprobs != "none" else None
            ents = [math.log(2)] * len(ids) if request.logprobs == "full" else None
            out.append(
                GenerationResult(
                    token_ids=ids,
                    text=t,
                    finish_reason="stop",
                    prompt_tokens=len(request.prompt_ids),
                    backend=self.name,
                    model=self.info.model,
                    tokenizer=self.info.tokenizer,
                    seed=request.seed,
                    params=to_plain(request.params),
                    logprob_mode=request.logprobs,
                    token_logprobs=lps,
                    token_entropies=ents,
                    scorer=self.scorer_id if lps is not None else None,
                )
            )
        return out

    def generate_batch(self, requests):
        return [self.generate(r) for r in requests]

    def score(self, prompt_ids, token_ids, full_vocabulary):
        self.score_calls.append((list(prompt_ids), list(token_ids), full_vocabulary))
        ents = [math.log(2)] * len(token_ids) if full_vocabulary else None
        return TokenScores([self.logprob] * len(token_ids), ents, self.scorer_id)

    def parse(self, text: str) -> ParsedOutput:
        return parse_qwen_output(text)

    def count_tokens(self, text: str) -> int:
        return len(text)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def scripted_backend_cls():
    return ScriptedBackend
