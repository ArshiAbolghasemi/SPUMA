import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml

from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, Settings, get_settings, load_settings
from spuma.config.settings import _redact


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
    with pytest.raises((ValueError, TypeError)):
        Settings.from_dict(raw_config)
    raw_config["model"]["backend"] = "vllm"
    raw_config["model"]["vllm"]["gpu_memory_utilization"] = 1.5
    with pytest.raises((ValueError, TypeError)):
        Settings.from_dict(raw_config)
    raw_config["model"]["vllm"]["gpu_memory_utilization"] = 0.8
    raw_config["model"]["vllm"]["max_logprobs"] = -2
    with pytest.raises((ValueError, TypeError)):
        Settings.from_dict(raw_config)


def test_cache_fingerprints_split_generation_and_scoring(config):
    hybrid = dataclasses.replace(
        config, model=dataclasses.replace(config.model, **{"backend": "vllm", "scoring_backend": "transformers"})
    )
    vllm_only = dataclasses.replace(
        config, model=dataclasses.replace(config.model, backend="vllm", scoring_backend="same")
    )
    # same generator, but `same` scores while generating and hybrid does not: different trajectories
    assert hybrid.generation_fingerprint() != vllm_only.generation_fingerprint()
    assert hybrid.scoring_fingerprint() != vllm_only.scoring_fingerprint()
    # changing only a scoring-engine setting keeps the generation cache
    chunked = dataclasses.replace(hybrid.model.transformers, **{"scoring_chunk_size": 64})
    tweaked = dataclasses.replace(hybrid, model=dataclasses.replace(hybrid.model, **{"transformers": chunked}))
    assert tweaked.generation_fingerprint() == hybrid.generation_fingerprint()
    assert tweaked.scoring_fingerprint() != hybrid.scoring_fingerprint()


def test_dynaconf_nested_env_override_is_typed(monkeypatch):
    monkeypatch.setenv("SPUMA_MODEL__GENERATION__TEMPERATURE", "0.2")
    monkeypatch.setenv("SPUMA_TOOLS__REGISTRY__MATH_VERIFY__ENABLED", "false")
    s = load_settings()
    assert s.model.generation.temperature == 0.2 and s.tools.registry["math_verify"].enabled is False


def test_get_settings_is_cached_until_cleared(monkeypatch):
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("SPUMA_DATASET__TEST_SAMPLES", "3")
    assert get_settings().dataset.test_samples == first.dataset.test_samples  # cached
    get_settings.cache_clear()
    assert get_settings().dataset.test_samples == 3


def test_settings_file_from_env(monkeypatch, tmp_path, raw_config):
    raw_config["experiment"]["name"] = "other_experiment"
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    assert load_settings().experiment.name == "other_experiment"


def test_unknown_and_missing_keys_name_the_path(raw_config):
    raw_config["tools"]["registry"]["sympy_solve"]["imaeg"] = "x"
    with pytest.raises(ValueError, match="tools.registry.sympy_solve.imaeg"):
        Settings.from_dict(raw_config)
    del raw_config["tools"]["registry"]["sympy_solve"]["imaeg"]
    del raw_config["agents"]["items"][0]["system_prompt"]
    with pytest.raises(ValueError, match=r"agents\.items\.0\.system_prompt"):
        Settings.from_dict(raw_config)


def test_unregistered_tool_rejected(raw_config):
    raw_config["agents"]["items"][0]["allowed_tools"].append("nonexistent_tool")
    with pytest.raises((ValueError, TypeError), match="not registered"):
        Settings.from_dict(raw_config)


def test_empty_allowed_tools_ok(raw_config):
    for a in raw_config["agents"]["items"]:
        a["allowed_tools"] = []
    Settings.from_dict(raw_config)


def test_feature_order_must_match_agents(raw_config):
    raw_config["features"]["order"] = raw_config["features"]["order"][:-1]
    with pytest.raises((ValueError, TypeError), match="features.order"):
        Settings.from_dict(raw_config)


def test_unknown_key_rejected(raw_config):
    raw_config["model"]["temprature"] = 1.0
    with pytest.raises((ValueError, TypeError)):
        Settings.from_dict(raw_config)


def test_semantic_entropy_is_uqlm_only(raw_config, config):
    se = config.uncertainty.semantic_entropy
    assert (
        se.backend == "uqlm"
        and se.nli_model_name == "tasksource/ModernBERT-large-nli"
        and se.nli_max_tokens == 2048
        and se.n == 10
    )
    for removed in ("native", "math_equivalence"):
        raw_config["uncertainty"]["semantic_entropy"]["backend"] = removed
        with pytest.raises((ValueError, TypeError)):
            Settings.from_dict(raw_config)


def test_changing_nli_model_keeps_generation_cache(config):
    se = dataclasses.replace(config.uncertainty.semantic_entropy, **{"nli_model_name": "other/nli"})
    other = dataclasses.replace(config, uncertainty=dataclasses.replace(config.uncertainty, **{"semantic_entropy": se}))
    assert other.generation_fingerprint() == config.generation_fingerprint()  # samples are reused
    assert other.scoring_fingerprint() != config.scoring_fingerprint()  # but re-clustered


def test_unknown_model_reference(raw_config):
    raw_config["agents"]["items"][1]["model"] = "other"
    with pytest.raises((ValueError, TypeError), match="models"):
        Settings.from_dict(raw_config)
    raw_config["models"] = {"other": {"name": "Qwen/Qwen3-0.6B"}}
    config = Settings.from_dict(raw_config)
    assert config.model_for(config.agents.items[1]).name == "Qwen/Qwen3-0.6B"


def test_sanitized_redacts_secrets(config):
    assert _redact({"a": {"api_key": "hunter2", "hf_token": "x", "max_output_tokens": 1}}) == {
        "a": {"api_key": "***", "hf_token": "***", "max_output_tokens": 1}
    }
    snap = config.sanitized()
    assert snap["tools"]["execution"]["max_output_tokens"] == 1024
    json.dumps(snap)  # JSON-able (paths as strings)


def test_dotenv_exported_not_in_config(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_secret\n")
    config = load_settings(dotenv_path=env)

    assert os.environ["HF_TOKEN"] == "hf_secret"
    assert "hf_secret" not in json.dumps(config.sanitized())


def test_only_one_yaml_file():

    found = [p for p in Path(PROJECT_ROOT).rglob("*.y*ml") if ".venv" not in p.parts and "artifacts" not in p.parts]
    assert [p.name for p in found] == ["config-qwen-math.yaml"], found
    assert yaml.safe_load(found[0].read_text())
