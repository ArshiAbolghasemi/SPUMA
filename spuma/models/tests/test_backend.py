import dataclasses

import pytest

from spuma.config import Settings
from spuma.models import engines as engines_module
from spuma.models import vllm_backend as vllm_module
from spuma.models.backend import (
    BackendCapabilities,
    BackendCapabilityError,
    generation_logprob_mode,
    require,
    scorer_id,
)
from spuma.models.config import ModelConfig, VLLMBackendConfig
from spuma.models.engines import get_backend, release_backends
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
    monkeypatch.setattr(engines_module, "_INSTANCES", {})
    monkeypatch.setitem(engines_module.BACKENDS, "transformers", _Fake)
    _Fake.made.clear()
    cfg = ModelConfig()
    a = get_backend(cfg, "transformers")
    b = get_backend(
        dataclasses.replace(cfg, **{"generation": dataclasses.replace(cfg.generation, **{"temperature": 0.1})})
    )
    assert a is b and len(_Fake.made) == 1
    release_backends()
    assert a.closed and not engines_module._INSTANCES


def test_require_and_missing():
    caps = BackendCapabilities(generation=True)
    assert caps.missing("generation", "teacher_forced_scoring") == ["teacher_forced_scoring"]
    backend = type("B", (), {"name": "b", "capabilities": caps})()
    with pytest.raises(BackendCapabilityError, match="teacher_forced_scoring"):
        require(backend, "teacher_forced_scoring")
    with pytest.raises(ValueError):
        caps.missing("teleport")


def test_fingerprints_separate_engines_and_generation():
    base = ModelConfig()
    assert base.fingerprint("transformers") != base.fingerprint("vllm")
    hybrid = dataclasses.replace(base, **{"backend": "vllm", "scoring_backend": "transformers"})
    assert hybrid.scoring_kind == "transformers" and not hybrid.scores_at_generation
    tweaked = dataclasses.replace(hybrid, **{"vllm": VLLMBackendConfig(gpu_memory_utilization=0.5)})
    assert tweaked.fingerprint("transformers") == hybrid.fingerprint("transformers")
    assert tweaked.generation_fingerprint() != hybrid.generation_fingerprint()
    assert scorer_id(base, "transformers").startswith("transformers:")


def test_generation_logprob_mode(config, set_settings):
    same = dataclasses.replace(config.model, backend="transformers", scoring_backend="same")
    assert generation_logprob_mode(same) == "full"
    hybrid = dataclasses.replace(config.model, **{"backend": "vllm", "scoring_backend": "transformers"})
    assert generation_logprob_mode(hybrid) == "none"
    uq = config.uncertainty
    no_pe = dataclasses.replace(
        uq, **{"predictive_entropy": dataclasses.replace(uq.predictive_entropy, **{"enabled": False})}
    )
    set_settings(dataclasses.replace(config, uncertainty=no_pe, features=_features_without(config, "_pe")))
    assert generation_logprob_mode(same) == "token"


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
    assert {"generation", "multiple_completions"} <= plan["generation_requires"]
    assert gen_req <= plan["generation_requires"] and plan["scoring_requires"] == score_req
    assert not ({"full_vocabulary_logprobs"} <= plan["generation_requires"]) or scoring == "same"


def test_validation_passes_for_transformers(with_model):
    with_model(backend="transformers", scoring_backend="same")
    plan = validate_backends()
    assert plan["default"]["generation_backend"] == "transformers"


def test_validation_fails_early_without_vllm(with_model, monkeypatch):
    monkeypatch.setattr(vllm_module, "SamplingParams", None)
    with_model(backend="vllm")
    with pytest.raises(BackendCapabilityError, match="vLLM is not installed"):
        validate_backends()


def test_validation_reports_capped_logprobs(with_model, set_settings, fake_vllm_module):
    capped = with_model(backend="vllm", scoring_backend="same", vllm={"max_logprobs": 20})
    with pytest.raises(BackendCapabilityError, match="full_vocabulary_logprobs"):
        validate_backends()
    raw = capped.to_dict()
    raw["uncertainty"]["predictive_entropy"]["enabled"] = False
    raw["features"]["order"] = [f for f in raw["features"]["order"] if not f.endswith("_pe")]
    set_settings(Settings.from_dict(raw))
    validate_backends()


def test_hybrid_validates_scoring_backend(with_model, fake_vllm_module):
    with_model(backend="vllm", scoring_backend="transformers")
    assert validate_backends()["default"]["scoring_backend"] == "transformers"
