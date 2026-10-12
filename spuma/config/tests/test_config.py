import json

import pytest
import yaml
from pydantic import ValidationError

from spuma.config import Settings, apply_env_overrides, load_settings


def test_project_config_is_valid(config):
    assert [a.id for a in config.agents.enabled] == ["solver", "reviewer"]
    assert config.model.name == "Qwen/Qwen3-8B"
    assert config.uncertainty.semantic_entropy.n == 10
    assert config.uncertainty.semantic_entropy.top_k is None
    assert len(config.features.order) == 8


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPUMA_MODEL__TRANSFORMERS__DEVICE", "cpu")
    monkeypatch.setenv("SPUMA_MODEL__BACKEND", "vllm")
    monkeypatch.setenv("SPUMA_MODEL__SCORING_BACKEND", "transformers")
    monkeypatch.setenv("SPUMA_DATASET__TEST_SAMPLES", "7")
    config = load_settings()
    assert config.model.transformers.device == "cpu"
    assert (config.model.backend, config.model.scoring_kind, config.model.scores_at_generation) == (
        "vllm",
        "transformers",
        False,
    )
    assert config.dataset.test_samples == 7


def test_backend_selection_validated(raw_config):
    raw_config["model"]["backend"] = "llamacpp"
    with pytest.raises(ValidationError):
        Settings.from_dict(raw_config)
    raw_config["model"]["backend"] = "vllm"
    raw_config["model"]["vllm"]["gpu_memory_utilization"] = 1.5
    with pytest.raises(ValidationError):
        Settings.from_dict(raw_config)
    raw_config["model"]["vllm"]["gpu_memory_utilization"] = 0.8
    raw_config["model"]["vllm"]["max_logprobs"] = -2
    with pytest.raises(ValidationError):
        Settings.from_dict(raw_config)


def test_cache_fingerprints_split_generation_and_scoring(config):
    hybrid = config.replace(
        model=config.model.model_copy(update={"backend": "vllm", "scoring_backend": "transformers"})
    )
    vllm_only = config.replace(model=config.model.model_copy(update={"backend": "vllm"}))
    # same generator, but `same` scores while generating and hybrid does not: different trajectories
    assert hybrid.generation_fingerprint() != vllm_only.generation_fingerprint()
    assert hybrid.scoring_fingerprint() != vllm_only.scoring_fingerprint()
    # changing only a scoring-engine setting keeps the generation cache
    chunked = hybrid.model.transformers.model_copy(update={"scoring_chunk_size": 64})
    tweaked = hybrid.replace(model=hybrid.model.model_copy(update={"transformers": chunked}))
    assert tweaked.generation_fingerprint() == hybrid.generation_fingerprint()
    assert tweaked.scoring_fingerprint() != hybrid.scoring_fingerprint()


def test_apply_env_overrides_nested():
    raw = {"model": {"generation": {"temperature": 0.7}}}
    applied = apply_env_overrides(raw, {"SPUMA_MODEL__GENERATION__TEMPERATURE": "0.2", "OTHER": "x"})
    assert raw["model"]["generation"]["temperature"] == 0.2
    assert applied == ["model.generation.temperature"]


def test_unregistered_tool_rejected(raw_config):
    raw_config["agents"]["items"][0]["allowed_tools"].append("nonexistent_tool")
    with pytest.raises(ValidationError, match="not registered"):
        Settings.from_dict(raw_config)


def test_empty_allowed_tools_ok(raw_config):
    for a in raw_config["agents"]["items"]:
        a["allowed_tools"] = []
    Settings.from_dict(raw_config)


def test_feature_order_must_match_agents(raw_config):
    raw_config["features"]["order"] = raw_config["features"]["order"][:-1]
    with pytest.raises(ValidationError, match="features.order"):
        Settings.from_dict(raw_config)


def test_unknown_key_rejected(raw_config):
    raw_config["model"]["temprature"] = 1.0
    with pytest.raises(ValidationError):
        Settings.from_dict(raw_config)


def test_semantic_entropy_is_uqlm_only(raw_config, config):
    se = config.uncertainty.semantic_entropy
    assert se.backend == "uqlm" and se.nli_model_name == "microsoft/deberta-large-mnli" and se.n == 10
    for removed in ("native", "math_equivalence"):
        raw_config["uncertainty"]["semantic_entropy"]["backend"] = removed
        with pytest.raises(ValidationError):
            Settings.from_dict(raw_config)


def test_changing_nli_model_keeps_generation_cache(config):
    se = config.uncertainty.semantic_entropy.model_copy(update={"nli_model_name": "other/nli"})
    other = config.replace(uncertainty=config.uncertainty.model_copy(update={"semantic_entropy": se}))
    assert other.generation_fingerprint() == config.generation_fingerprint()  # samples are reused
    assert other.scoring_fingerprint() != config.scoring_fingerprint()  # but re-clustered


def test_unknown_model_reference(raw_config):
    raw_config["agents"]["items"][1]["model"] = "other"
    with pytest.raises(ValidationError, match="models"):
        Settings.from_dict(raw_config)
    raw_config["models"] = {"other": {"name": "Qwen/Qwen3-0.6B"}}
    config = Settings.from_dict(raw_config)
    assert config.model_for(config.agents.items[1]).name == "Qwen/Qwen3-0.6B"


def test_variants_override_allowed_tools(config):
    no_tools = config.with_variant("no_tools")
    assert all(a.allowed_tools == [] for a in no_tools.agents.items)
    solver_only = config.with_variant("solver_tools")
    assert solver_only.agents.items[0].allowed_tools == config.agents.items[0].allowed_tools
    assert solver_only.agents.items[1].allowed_tools == []


def test_sanitized_redacts_secrets(raw_config):
    raw_config["tools"]["registry"]["sympy_solve"]["api_key"] = "hunter2"
    snap = Settings.from_dict(raw_config).sanitized()
    assert snap["tools"]["registry"]["sympy_solve"]["api_key"] == "***"
    assert snap["tools"]["execution"]["max_output_tokens"] == 1024
    assert "hunter2" not in json.dumps(snap)


def test_dotenv_exported_not_in_config(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_secret\n")
    config = load_settings(dotenv_path=env)
    import os

    assert os.environ["HF_TOKEN"] == "hf_secret"
    assert "hf_secret" not in json.dumps(config.sanitized())


def test_only_one_yaml_file():
    from pathlib import Path

    from spuma.config import PROJECT_ROOT

    found = [p for p in Path(PROJECT_ROOT).rglob("*.y*ml") if ".venv" not in p.parts and "artifacts" not in p.parts]
    assert [p.name for p in found] == ["config-qwen-math.yaml"], found
    assert yaml.safe_load(found[0].read_text())
