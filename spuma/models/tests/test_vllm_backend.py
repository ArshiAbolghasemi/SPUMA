"""vLLM backend logic against the fake `vllm` module (real raw log-probs from a tiny HF model)."""

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from langchain_core.messages import HumanMessage

from spuma.models import vllm_backend as vllm_module
from spuma.models.backend import BackendCapabilityError, GenerationRequest, ModelBackend
from spuma.models.config import GenerationConfig, ModelConfig, VLLMBackendConfig
from spuma.models.tests import fake_vllm
from spuma.models.vllm_backend import VLLMBackend, _positions

SAMPLE = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=6)
GREEDY = GenerationConfig(temperature=0.0, max_new_tokens=6)


def _prompt(b):
    return b.render([HumanMessage("Solve 2x+3=7")])


def test_capabilities_require_installed_vllm(monkeypatch):
    monkeypatch.setattr(vllm_module, "SamplingParams", None)
    with pytest.raises(BackendCapabilityError, match="not installed"):
        VLLMBackend.capabilities_for(ModelConfig(backend="vllm"))


def test_capabilities_follow_max_logprobs(fake_vllm_module):
    full = VLLMBackend.capabilities_for(ModelConfig(backend="vllm"))
    assert full.full_vocabulary_logprobs and full.full_vocabulary_scoring and full.multiple_completions
    capped = VLLMBackend.capabilities_for(ModelConfig(backend="vllm", vllm=VLLMBackendConfig(max_logprobs=20)))
    assert not capped.full_vocabulary_logprobs and capped.generated_token_logprobs and capped.teacher_forced_scoring


def test_capabilities_reject_incompatible_version(fake_vllm_module, monkeypatch):
    class Old:
        __dataclass_fields__ = {"n": None, "seed": None}

    monkeypatch.setattr(vllm_module, "SamplingParams", Old)
    with pytest.raises(BackendCapabilityError, match="lacks SamplingParams fields"):
        VLLMBackend.capabilities_for(ModelConfig(backend="vllm"))


def test_sampling_params_translation(vllm_backend):
    sp = vllm_backend.sampling_params(SAMPLE, n=10, seed=3, logprobs="full")
    assert (sp.n, sp.temperature, sp.top_p, sp.top_k, sp.seed) == (10, 0.7, 0.9, 0, 3)
    assert sp.logprobs == -1 and sp.flat_logprobs and not sp.detokenize and sp.max_tokens == 6
    assert vllm_backend.sampling_params(SAMPLE, 1, 0, "token").logprobs == 0
    assert vllm_backend.sampling_params(SAMPLE, 1, 0, "none").logprobs is None
    assert vllm_backend.sampling_params(GREEDY, 1, 0, "none").temperature == 0.0
    with pytest.raises(ValueError):
        vllm_backend.sampling_params(GREEDY, 2, 0, "none")


def test_full_logprobs_need_uncapped_engine(vllm_backend):
    vllm_backend.config = dataclasses.replace(vllm_backend.config, **{"vllm": VLLMBackendConfig(max_logprobs=20)})
    with pytest.raises(BackendCapabilityError, match="max_logprobs"):
        vllm_backend.sampling_params(SAMPLE, 1, 0, "full")


def test_protocol_and_schema(vllm_backend):
    assert isinstance(vllm_backend, ModelBackend)
    prompt = _prompt(vllm_backend)
    (gen,) = vllm_backend.generate(GenerationRequest(prompt, SAMPLE, seed=1, logprobs="full"))
    assert gen.backend == "vllm" and gen.prompt_tokens == len(prompt) and gen.finish_reason in ("stop", "length")
    assert (
        len(gen.token_logprobs) == len(gen.token_entropies) == gen.num_tokens and gen.scorer == vllm_backend.scorer_id
    )
    assert gen.text == vllm_backend.chat.decode(gen.token_ids)


def test_generation_logprobs_are_raw_model(vllm_backend, tiny_backend):
    """Generated-token log-probs/entropies equal Transformers teacher-forced scores of the same tokens."""
    prompt = _prompt(vllm_backend)
    (gen,) = vllm_backend.generate(GenerationRequest(prompt, SAMPLE, seed=4, logprobs="full"))
    ref = tiny_backend.score(prompt, gen.token_ids, full_vocabulary=True)
    assert gen.token_logprobs == pytest.approx(ref.token_logprobs, abs=1e-5)
    assert gen.token_entropies == pytest.approx(ref.token_entropies, abs=1e-5)


def test_batch_and_ten_samples(vllm_backend):
    prompt = _prompt(vllm_backend)
    batch = vllm_backend.generate_batch(
        [GenerationRequest(prompt, SAMPLE, seed=5, n=10), GenerationRequest(prompt, GREEDY, seed=0, logprobs="token")]
    )
    assert [len(r) for r in batch] == [10, 1]
    assert all(r.token_logprobs is None for r in batch[0]) and batch[1][0].token_entropies is None
    assert len(vllm_backend.llm.requests) == 2


def test_teacher_forced_score_alignment(vllm_backend, tiny_backend):
    prompt = _prompt(vllm_backend)
    continuation = prompt[2:9]
    v = vllm_backend.score(prompt, continuation, full_vocabulary=True)
    t = tiny_backend.score(prompt, continuation, full_vocabulary=True)
    assert v.token_logprobs == pytest.approx(t.token_logprobs, abs=1e-5)
    assert v.token_entropies == pytest.approx(t.token_entropies, abs=1e-5)
    nll_only = vllm_backend.score(prompt, continuation, full_vocabulary=False)
    assert nll_only.token_entropies is None and nll_only.token_logprobs == pytest.approx(t.token_logprobs, abs=1e-5)
    _, sp = vllm_backend.llm.requests[-1]
    assert sp.max_tokens == 1 and sp.prompt_logprobs == 1


def test_engine_calls_from_several_threads_are_serialized(vllm_backend, monkeypatch):
    """vLLM's offline LLM mixes up the outputs of concurrent calls, e.g. a worker scoring while the batcher generates."""
    generate, active, overlaps = vllm_backend.llm.generate, [], []

    def exclusive(*args, **kwargs):
        overlaps.append(bool(active))
        active.append(1)
        time.sleep(0.01)
        try:
            return generate(*args, **kwargs)
        finally:
            active.pop()

    monkeypatch.setattr(vllm_backend.llm, "generate", exclusive)
    prompt = _prompt(vllm_backend)
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(vllm_backend.score, prompt, prompt[2:9], False) for _ in range(4)]
        futures += [pool.submit(vllm_backend.generate, GenerationRequest(prompt, GREEDY, seed=0)) for _ in range(4)]
        [f.result() for f in futures]
    assert len(overlaps) == 8 and not any(overlaps)


def test_score_respects_max_model_len(vllm_backend):
    vllm_backend.config = dataclasses.replace(vllm_backend.config, **{"vllm": VLLMBackendConfig(max_model_len=4)})
    with pytest.raises(BackendCapabilityError, match="max_model_len"):
        vllm_backend.score([1, 2, 3], [4, 5], full_vocabulary=False)


def test_truncated_distribution_is_rejected(vllm_backend):
    flat = fake_vllm.FlatLogprobs([0], [3], [5, 5, 7], [-0.1, -0.1, -2.0])
    with pytest.raises(BackendCapabilityError, match="truncated entropy"):
        vllm_backend._extract(_positions(flat), [5], full=True)


def test_missing_token_logprob_is_rejected(vllm_backend):
    flat = fake_vllm.FlatLogprobs([0], [1], [7], [-2.0])
    with pytest.raises(BackendCapabilityError, match="missing"):
        vllm_backend._extract(_positions(flat), [5], full=False)


def test_dict_logprob_format_supported():
    lp = type("L", (), {})

    def mk(v):
        o = lp()
        o.logprob = v
        return o

    positions = list(_positions([None, {3: mk(-0.5), 4: mk(-1.0)}]))
    assert (
        len(positions[0][0]) == 0 and positions[1][0].tolist() == [3, 4] and np.allclose(positions[1][1], [-0.5, -1.0])
    )


def test_close(vllm_backend):
    vllm_backend.close()
    assert vllm_backend.llm is None
