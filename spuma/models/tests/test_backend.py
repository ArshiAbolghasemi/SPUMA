import dataclasses

import pytest

from spuma.config import Settings
from spuma.models import backend as backend_module
from spuma.models.backend import (
    BackendCapabilities,
    BackendCapabilityError,
    generation_logprob_mode,
    get_backend,
    release_backends,
    require,
    scorer_id,
)
from spuma.models.config import ModelConfig, VLLMBackendConfig
from spuma.models.validation import execution_plan, validate_backends


class _Fake:
    made = []

    def __init__(self, config):
        self.config, self.closed = config, False
        self.name = "fake"
        self.capabilities = BackendCapabilities()

    @staticmethod
    def capabilities_for(config):
        return BackendCapabilities()

    @classmethod
    def from_config(cls, config):
        cls.made.append(config)
        return cls(config)

    def close(self):
        self.closed = True


def test_engine_reuse_and_release(monkeypatch):
    monkeypatch.setattr(backend_module, "_INSTANCES", {})
    monkeypatch.setitem(backend_module._CLASSES, "transformers", _Fake)
    _Fake.made.clear()
    cfg = ModelConfig()
    a = get_backend(cfg, "transformers")
    b = get_backend(
        dataclasses.replace(cfg, **{"generation": dataclasses.replace(cfg.generation, **{"temperature": 0.1})})
    )
    assert a is b and len(_Fake.made) == 1  # generation settings do not create a new engine
    release_backends()
    assert a.closed and not backend_module._INSTANCES


def test_require_and_missing():
    caps = BackendCapabilities(generation=True)
    assert caps.missing("generation", "tool_calling") == ["tool_calling"]
    backend = type("B", (), {"name": "b", "capabilities": caps})()
    with pytest.raises(BackendCapabilityError, match="tool_calling"):
        require(backend, "tool_calling")
    with pytest.raises(ValueError):
        caps.missing("teleport")


def test_fingerprints_separate_engines_and_generation():
    base = ModelConfig()
    assert base.fingerprint("transformers") != base.fingerprint("vllm")
    hybrid = dataclasses.replace(base, **{"backend": "vllm", "scoring_backend": "transformers"})
    assert hybrid.scoring_kind == "transformers" and not hybrid.scores_at_generation
    # vLLM engine settings do not touch the Transformers engine identity, and vice versa
    tweaked = dataclasses.replace(hybrid, **{"vllm": VLLMBackendConfig(gpu_memory_utilization=0.5)})
    assert tweaked.fingerprint("transformers") == hybrid.fingerprint("transformers")
    assert tweaked.generation_fingerprint() != hybrid.generation_fingerprint()
    assert scorer_id(base, "transformers").startswith("transformers:")


def test_generation_logprob_mode(config, set_settings):
    assert generation_logprob_mode(config.model) == "full"
    hybrid = dataclasses.replace(config.model, **{"backend": "vllm", "scoring_backend": "transformers"})
    assert generation_logprob_mode(hybrid) == "none"
    uq = config.uncertainty
    no_pe = dataclasses.replace(
        uq, **{"predictive_entropy": dataclasses.replace(uq.predictive_entropy, **{"enabled": False})}
    )
    set_settings(dataclasses.replace(config, uncertainty=no_pe, features=_features_without(config, "_pe")))
    assert generation_logprob_mode(config.model) == "token"


def _features_without(config, suffix):
    return dataclasses.replace(
        config.features, **{"order": [f for f in config.features.order if not f.endswith(suffix)]}
    )


@pytest.fixture
def with_model(config, set_settings):
    """Install settings whose `model:` section has these fields replaced."""

    def install(**model):
        raw = config.to_dict()
        raw["model"].update(model)
        return set_settings(Settings.from_dict(raw))

    return install


@pytest.mark.parametrize(
    ("backend", "scoring", "gen_req", "score_req"),
    [
        ("transformers", "same", {"full_vocabulary_logprobs", "generated_token_logprobs"}, set()),
        ("vllm", "same", {"full_vocabulary_logprobs", "generated_token_logprobs"}, set()),
        ("vllm", "transformers", set(), {"teacher_forced_scoring", "full_vocabulary_scoring"}),
    ],
)
def test_execution_plan_modes(with_model, backend, scoring, gen_req, score_req):
    with_model(backend=backend, scoring_backend=scoring)
    plan = execution_plan()["default"]
    assert plan["agents"] == ["solver", "reviewer"]
    assert {"generation", "tool_calling", "multiple_completions"} <= plan["generation_requires"]
    assert gen_req <= plan["generation_requires"] and plan["scoring_requires"] == score_req
    assert not ({"full_vocabulary_logprobs"} <= plan["generation_requires"]) or scoring == "same"


def test_execution_plan_per_variant():
    # no_tools: neither agent may call tools, so tool calling is not required for that variant
    assert "tool_calling" not in execution_plan("no_tools")["default"]["generation_requires"]
    assert "tool_calling" in execution_plan("both_tools")["default"]["generation_requires"]


def test_validation_passes_for_transformers():
    plan = validate_backends()
    assert plan["default"]["generation_backend"] == "transformers"


def test_validation_fails_early_without_vllm(with_model, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with_model(backend="vllm")
    with pytest.raises(BackendCapabilityError, match="vLLM is not installed"):
        validate_backends()


def test_validation_reports_capped_logprobs(with_model, set_settings, fake_vllm_module):
    capped = with_model(backend="vllm", vllm={"max_logprobs": 20})
    with pytest.raises(BackendCapabilityError, match="full_vocabulary_logprobs"):
        validate_backends()
    # NLL-only experiments do not need the full vocabulary
    raw = capped.to_dict()
    raw["uncertainty"]["predictive_entropy"]["enabled"] = False
    raw["features"]["order"] = [f for f in raw["features"]["order"] if not f.endswith("_pe")]
    set_settings(Settings.from_dict(raw))
    validate_backends()


def test_hybrid_validates_scoring_backend(with_model, fake_vllm_module):
    with_model(backend="vllm", scoring_backend="transformers")
    assert validate_backends()["default"]["scoring_backend"] == "transformers"
