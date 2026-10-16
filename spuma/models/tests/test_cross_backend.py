"""Cross-backend numerical validation on fixed prompt/continuation token sequences.

Offline: the vLLM code path (FlatLogprobs extraction, dedupe, row alignment) on the fake engine must
reproduce Transformers scores of the same tiny model exactly (float32, atol 1e-5).

Real (`-m vllm`, Linux + GPU): Transformers vs vLLM on Qwen3-0.6B in float32 for fixed sequences.
Documented tolerances (per token, nats): |Δ log p| <= 2e-3 and |Δ H| <= 5e-3; step means
|ΔNLL|, |ΔPE| <= 1e-3. Differences come from kernel/accumulation order; anything larger fails.
"""

import logging

import numpy as np
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from spuma.models.backend import GenerationRequest
from spuma.models.config import GenerationConfig, ModelConfig, TransformersBackendConfig, VLLMBackendConfig
from spuma.models.engines import get_backend, release_backends
from spuma.uncertainty.aggregation import rms

logger = logging.getLogger(__name__)

TOKEN_ATOL_LOGP, TOKEN_ATOL_H, STEP_ATOL = 2e-3, 5e-3, 1e-3


def compare(t_backend, v_backend, prompt_ids, continuation):
    t = t_backend.score(prompt_ids, continuation, full_vocabulary=True)
    v = v_backend.score(prompt_ids, continuation, full_vocabulary=True)
    d_lp = np.abs(np.array(t.token_logprobs) - np.array(v.token_logprobs))
    d_h = np.abs(np.array(t.token_entropies) - np.array(v.token_entropies))
    stats = {
        "tokens": len(continuation),
        "max_dlogp": float(d_lp.max()),
        "max_dH": float(d_h.max()),
        "dNLL": abs(np.mean(t.token_logprobs) - np.mean(v.token_logprobs)),
        "dPE": abs(np.mean(t.token_entropies) - np.mean(v.token_entropies)),
    }
    logger.info("cross-backend scores", extra=stats)
    return t, v, stats


def test_offline_vllm_path_matches_transformers(tiny_backend, vllm_backend):
    msgs = [SystemMessage("sys"), HumanMessage("Solve 2x+3=7")]
    prompt = tiny_backend.render(msgs)
    assert vllm_backend.render(msgs) == prompt  # identical tokenization and chat template
    assert vllm_backend.info.compatibility_key() == tiny_backend.info.compatibility_key()
    continuation = tiny_backend.chat.encode("The answer is \\boxed{2}.") + [tiny_backend.chat.tokenizer.eos_token_id]
    t, v, stats = compare(tiny_backend, vllm_backend, prompt, continuation)
    assert stats["max_dlogp"] < 1e-5 and stats["max_dH"] < 1e-5
    # aggregation over steps is consistent too
    assert rms([np.mean(t.token_entropies), 1.0]) == pytest.approx(rms([np.mean(v.token_entropies), 1.0]), abs=1e-6)


@pytest.mark.vllm
def test_real_vllm_matches_transformers_qwen3_small():

    cfg = ModelConfig(
        name="Qwen/Qwen3-0.6B",
        dtype="float32",
        transformers=TransformersBackendConfig(device="cuda"),
        vllm=VLLMBackendConfig(max_model_len=1024, gpu_memory_utilization=0.4, enforce_eager=True),
    )
    msgs = [SystemMessage("Solve step by step."), HumanMessage("What is 12 * 7?")]
    vllm_b = get_backend(cfg, "vllm")
    prompt = vllm_b.render(msgs)
    continuation = vllm_b.chat.encode("12 * 7 = 84. The answer is \\boxed{84}.")
    v = vllm_b.score(prompt, continuation, full_vocabulary=True)
    release_backends("vllm")
    tf_b = get_backend(cfg, "transformers")
    assert tf_b.render(msgs) == prompt and tf_b.info.compatibility_key() == vllm_b.info.compatibility_key()
    t = tf_b.score(prompt, continuation, full_vocabulary=True)
    d_lp = np.abs(np.array(t.token_logprobs) - np.array(v.token_logprobs))
    d_h = np.abs(np.array(t.token_entropies) - np.array(v.token_entropies))
    logger.info("real cross-backend", extra={"max_dlogp": float(d_lp.max()), "max_dH": float(d_h.max())})
    assert d_lp.max() <= TOKEN_ATOL_LOGP and d_h.max() <= TOKEN_ATOL_H
    assert abs(np.mean(t.token_logprobs) - np.mean(v.token_logprobs)) <= STEP_ATOL
    assert abs(np.mean(t.token_entropies) - np.mean(v.token_entropies)) <= STEP_ATOL
    release_backends()


@pytest.mark.vllm
def test_real_vllm_qwen3_8b_sampling_and_logprobs():

    backend = get_backend(ModelConfig(backend="vllm"), "vllm")
    prompt = backend.render([HumanMessage("What is 2+2? Answer with \\boxed{}.")])
    params = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=64)
    samples = backend.generate(GenerationRequest(prompt, params, seed=0, n=10))
    assert len(samples) == 10
    (main,) = backend.generate(GenerationRequest(prompt, params, seed=1, logprobs="full"))
    assert len(main.token_entropies) == main.num_tokens and all(lp <= 0 for lp in main.token_logprobs)
    release_backends()
