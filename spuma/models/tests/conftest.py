import dataclasses

import pytest

from spuma.models import vllm_backend as vllm_module
from spuma.models.tests import fake_vllm
from spuma.models.tests.fixtures import TINY_CONFIG
from spuma.models.vllm_backend import VLLMBackend


@pytest.fixture
def fake_vllm_module(monkeypatch):
    """Make the vLLM backend see the fake engine as the installed `vllm` (it imports vLLM at the top)."""
    m, inputs = fake_vllm.module()
    monkeypatch.setattr(vllm_module, "SamplingParams", m.SamplingParams)
    monkeypatch.setattr(vllm_module, "LLM", m.LLM)
    monkeypatch.setattr(vllm_module, "TokensPrompt", inputs.TokensPrompt)
    return m


@pytest.fixture
def vllm_backend(fake_vllm_module, tiny_model, tokenizer):
    llm = fake_vllm.LLM(tiny_model, [tokenizer.eos_token_id])
    config = dataclasses.replace(TINY_CONFIG, **{"backend": "vllm"})
    return VLLMBackend(llm, tokenizer, config, fake_vllm.SamplingParams, fake_vllm.TokensPrompt)
