import dataclasses
import math

import pytest
import torch
from langchain_core.messages import HumanMessage, SystemMessage

from spuma.models.backend import GenerationRequest, GenerationResult, ModelBackend
from spuma.models.config import GenerationConfig, ModelConfig
from spuma.models.engines import get_backend
from spuma.models.tests.fixtures import TINY_CONFIG
from spuma.models.transformers_backend import TransformersBackend

GREEDY = GenerationConfig(temperature=0.0, max_new_tokens=8)
SAMPLE = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=8)


def _prompt(backend):
    return backend.render([SystemMessage("sys"), HumanMessage("Solve 2x+3=7")], None)


def test_protocol_and_info(tiny_backend):
    assert isinstance(tiny_backend, ModelBackend)
    assert tiny_backend.info.backend == "transformers" and tiny_backend.info.model == "tiny-qwen3"
    assert tiny_backend.capabilities.full_vocabulary_scoring and tiny_backend.capabilities.tool_calling


def test_render_uses_chat_template(tiny_backend, tokenizer):
    text = tokenizer.decode(_prompt(tiny_backend))
    assert text.startswith("<|im_start|>system\nsys<|im_end|>") and text.endswith("<|im_start|>assistant\n")


def test_render_includes_tool_schemas(tiny_backend, tokenizer):
    tools = [{"type": "function", "function": {"name": "sympy_solve", "parameters": {}}}]
    assert "sympy_solve" in tokenizer.decode(tiny_backend.render([HumanMessage("q")], tools))


def test_greedy_matches_hf_generate(tiny_backend, tiny_model):
    prompt = _prompt(tiny_backend)
    (gen,) = tiny_backend.generate(GenerationRequest(prompt, GREEDY, seed=0))
    ref = tiny_model.generate(torch.tensor([prompt]), max_new_tokens=8, do_sample=False, pad_token_id=0, top_k=0)
    assert gen.token_ids == ref[0, len(prompt) :].tolist()[: len(gen.token_ids)]


def test_result_schema(tiny_backend):
    prompt = _prompt(tiny_backend)
    (gen,) = tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=1, logprobs="full"))
    assert isinstance(gen, GenerationResult)
    assert gen.prompt_tokens == len(prompt) and gen.num_tokens == len(gen.token_ids)
    assert gen.finish_reason in ("stop", "length") and gen.backend == "transformers" and gen.seed == 1
    assert gen.params["temperature"] == 0.7 and gen.scorer == tiny_backend.scorer_id
    assert set(gen.metadata()) >= {"finish_reason", "prompt_tokens", "num_tokens", "latency_s", "scorer"}


def test_logprob_modes(tiny_backend):
    prompt = _prompt(tiny_backend)
    none, token, full = (
        tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=1, logprobs=m))[0]
        for m in ("none", "token", "full")
    )
    assert none.token_logprobs is None and none.scorer is None
    assert token.token_logprobs is not None and token.token_entropies is None
    assert full.token_ids == token.token_ids and full.token_logprobs == pytest.approx(token.token_logprobs)
    assert len(full.token_entropies) == full.num_tokens


def test_generation_scores_are_raw_teacher_forced(tiny_backend, tiny_model):
    """Scores of a temperature-0.7 sample are the untempered model distribution at each position."""
    prompt = _prompt(tiny_backend)
    (gen,) = tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=1, logprobs="full"))
    full = torch.log_softmax(tiny_model(torch.tensor([prompt + gen.token_ids])).logits[0].float(), -1)
    for i, tok in enumerate(gen.token_ids):
        row = full[len(prompt) - 1 + i]
        assert gen.token_logprobs[i] == pytest.approx(row[tok].item(), abs=1e-4)
        assert gen.token_entropies[i] == pytest.approx(-(row.exp() * row).sum().item(), abs=1e-4)
        assert 0 <= gen.token_entropies[i] <= math.log(row.numel()) + 1e-4


def test_score_fixed_continuation(tiny_backend):
    prompt = _prompt(tiny_backend)
    (gen,) = tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=2, logprobs="full"))
    scores = tiny_backend.score(prompt, gen.token_ids, full_vocabulary=True)
    assert scores.token_logprobs == pytest.approx(gen.token_logprobs)
    assert scores.token_entropies == pytest.approx(gen.token_entropies)
    assert tiny_backend.score(prompt, gen.token_ids, full_vocabulary=False).token_entropies is None
    assert tiny_backend.score(prompt, [], full_vocabulary=True).token_logprobs == []


def test_chunked_scoring_matches(tiny_backend):
    prompt = _prompt(tiny_backend)
    tokens = prompt[:7]
    whole = tiny_backend.score(prompt, tokens, True)
    chunked = dataclasses.replace(tiny_backend.config.transformers, scoring_chunk_size=2)
    tiny_backend.config = dataclasses.replace(tiny_backend.config, transformers=chunked)
    chunked = tiny_backend.score(prompt, tokens, True)
    assert chunked.token_entropies == pytest.approx(whole.token_entropies, abs=1e-5)


def test_sampling_reproducible_and_batched(tiny_backend):
    prompt = _prompt(tiny_backend)
    a = tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=7, n=10))
    b = tiny_backend.generate(GenerationRequest(prompt, SAMPLE, seed=7, n=10))
    assert len(a) == 10 and [g.token_ids for g in a] == [g.token_ids for g in b]
    batch = tiny_backend.generate_batch(
        [GenerationRequest(prompt, SAMPLE, seed=7, n=10), GenerationRequest(prompt, GREEDY, 0)]
    )
    assert [len(r) for r in batch] == [10, 1] and [g.token_ids for g in batch[0]] == [g.token_ids for g in a]


def test_top_k_disabled_by_default(tiny_backend, mocker):
    spy = mocker.spy(tiny_backend.model, "generate")
    tiny_backend.generate(GenerationRequest(_prompt(tiny_backend), SAMPLE, seed=0))
    assert spy.call_args.kwargs["top_k"] == 0


def test_n_samples_need_temperature(tiny_backend):
    with pytest.raises(ValueError):
        tiny_backend.generate(GenerationRequest(_prompt(tiny_backend), GREEDY, seed=0, n=2))


def test_token_count_and_truncate(tiny_backend):
    assert tiny_backend.count_tokens("Solve for x") > 0
    assert tiny_backend.truncate("Solve for x " * 50, 5).endswith("[truncated]")


def test_close_releases_model(tiny_model, tokenizer):

    backend = TransformersBackend(tiny_model, tokenizer, TINY_CONFIG)
    backend.close()
    assert not hasattr(backend, "model")


@pytest.mark.gpu
def test_qwen3_real_generation():
    backend = get_backend(ModelConfig(), "transformers")
    prompt = backend.render([HumanMessage("What is 2+2? Answer with \\boxed{}.")], None)
    (gen,) = backend.generate(
        GenerationRequest(prompt, GenerationConfig(temperature=0.0, max_new_tokens=64), 0, logprobs="full")
    )
    assert "4" in gen.text and len(gen.token_entropies) == gen.num_tokens
