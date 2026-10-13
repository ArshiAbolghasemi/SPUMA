"""Shared fixtures for every `spuma/**/tests` package.

Offline only: a byte-level BPE tokenizer trained in-process, a tiny randomly initialised Qwen3
causal LM (no downloads, CPU), and a scripted backend that replays fixed outputs for workflow tests.
"""

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from langchain_core.messages import BaseMessage
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from spuma.common.config import to_plain
from spuma.config import Settings, get_settings
from spuma.config import settings as settings_module
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
    '<tool_call>{"name": "sympy_solve", "arguments": {"equations": ["x**2 - 4"]}}</tool_call>',
    "Review the solution, verify calculations and give the final answer.",
    "<|im_start|> <|im_end|> system user assistant tool <think> </think>",
] * 20

CHAT_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n# Tools\n{% for t in tools %}{{ t | tojson }}\n{% endfor %}<|im_end|>\n{% endif %}"
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Never read the project .env or inherit SPUMA_* / HF_TOKEN from the shell."""
    monkeypatch.setattr(settings_module, "DEFAULT_DOTENV_FILE", tmp_path / "no.env")
    for var in [v for v in os.environ if v.startswith("SPUMA_")] + ["HF_TOKEN"]:
        monkeypatch.setenv(var, "")  # record the original so teardown restores it (load_dotenv writes os.environ)
        monkeypatch.delenv(var)
    get_settings.cache_clear()  # every test starts from a fresh load of the real config file
    yield
    get_settings.cache_clear()


@pytest.fixture
def config() -> Settings:
    """The process-wide settings (`get_settings()`): the real config/config-qwen-math.yaml."""
    return get_settings()


@pytest.fixture
def set_settings(monkeypatch):
    """Make `get_settings()` return the given settings for this test (code reads settings, never receives them)."""

    def install(settings: Settings) -> Settings:
        monkeypatch.setattr(settings_module, "load_settings", lambda *args, **kwargs: settings)
        get_settings.cache_clear()
        return settings

    return install


@pytest.fixture
def raw_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(settings_module.DEFAULT_SETTINGS_FILE).read_text())


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
    tool_calling=True,
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
        self.tools_seen: list[list[str]] = []
        self.calls: list[GenerationRequest] = []
        self.score_calls: list[tuple[list[int], list[int], bool]] = []
        self.closed = False

    def render(self, messages: Sequence[BaseMessage], tools: list[dict[str, Any]] | None) -> list[int]:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_chat_dicts(messages))
        self.prompts.append(text)
        self.tools_seen.append([t["function"]["name"] for t in tools or []])
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

    def truncate(self, text: str, max_tokens: int) -> str:
        return text[:max_tokens]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def scripted_backend_cls():
    return ScriptedBackend


HEAVY_MARKERS = ("gpu", "vllm", "nli")


def pytest_collection_modifyitems(config, items):
    """Model-downloading tests (gpu, vllm, nli) run only when explicitly selected: their marker is named
    positively in `-m` (e.g. `-m nli`, `-m "gpu or vllm"`), or SPUMA_TESTS_ALL=1 (run-tests.sh --all).

    Any other invocation, including a bare `pytest` or a `-m` filter that simply does not mention a
    marker, skips them, so no test run downloads a model by accident.
    """
    import re

    if os.environ.get("SPUMA_TESTS_ALL") == "1":
        return
    expr = config.option.markexpr or ""
    selected = {m for m in HEAVY_MARKERS if re.search(rf"(?<!not )\b{m}\b", expr)}
    for item in items:
        heavy = [m for m in HEAVY_MARKERS if m in item.keywords]
        if heavy and not selected.intersection(heavy):
            reason = f"downloads a model ({', '.join(heavy)}): select with -m {heavy[0]} or run-tests.sh --all"
            item.add_marker(pytest.mark.skip(reason=reason))


class RuleNLI:
    """Stand-in for uqlm.nli.NLI with the same interface: "entailment" iff both texts end in the same number.

    UQLM's BlackBoxUQ, clustering and entropy code run unchanged on top of it; only the DeBERTa download
    is avoided.
    """

    def __init__(self):
        self.num_responses = None
        self.probabilities = {}
        self.pairs: list[tuple[str, str]] = []

    @staticmethod
    def _final_number(text: str):
        import re

        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[-1]) if nums else None

    def get_nli_results_batch(self, response_pairs):
        self.pairs += list(response_pairs)
        out = []
        for a, b in response_pairs:
            same = a == b or (self._final_number(a) is not None and self._final_number(a) == self._final_number(b))
            out.append(
                {"noncontradiction_score": 1.0 if same else 0.0, "entailment": same, "entailment_score": float(same)}
            )
        return out


@pytest.fixture(autouse=True)
def rule_nli(request, monkeypatch):
    """Offline tests never load the real NLI model; `nli`-marked tests do."""
    if request.node.get_closest_marker("nli"):
        return None
    nli = RuleNLI()
    from spuma.uncertainty import estimators

    real = estimators.load_black_box_uq

    def load():
        import time
        import types

        import uqlm.scorers.shortform.black_box as black_box
        import uqlm.scorers.shortform.entropy as entropy

        # UQLM's SemanticEntropy.score ends with a fixed time.sleep(0.1); skip it in tests only
        monkeypatch.setattr(entropy, "time", types.SimpleNamespace(sleep=lambda s: None, time=time.time))
        with pytest.MonkeyPatch.context() as m:  # real BlackBoxUQ, rule NLI instead of the DeBERTa download
            m.setattr(black_box, "NLI", lambda **kwargs: nli)
            return real()

    monkeypatch.setattr(estimators, "load_black_box_uq", load)
    return nli
