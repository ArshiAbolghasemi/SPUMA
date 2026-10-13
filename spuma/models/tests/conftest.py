import dataclasses
import sys

import pytest

from spuma.conftest import TINY_CONFIG
from spuma.models.tests import fake_vllm
from spuma.models.vllm_backend import VLLMBackend


@pytest.fixture
def fake_vllm_module(monkeypatch):
    m, inputs = fake_vllm.module()
    monkeypatch.setitem(sys.modules, "vllm", m)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    return m


@pytest.fixture
def vllm_backend(fake_vllm_module, tiny_model, tokenizer):
    llm = fake_vllm.LLM(tiny_model, [tokenizer.eos_token_id])
    config = dataclasses.replace(TINY_CONFIG, **{"backend": "vllm"})
    return VLLMBackend(llm, tokenizer, config, fake_vllm.SamplingParams, fake_vllm.TokensPrompt)
