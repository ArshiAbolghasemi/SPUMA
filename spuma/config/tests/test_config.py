import dataclasses
import json
import os
from pathlib import Path

import pytest
import yaml

from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, Settings, get_settings, load_settings
from spuma.config.settings import _redact
from spuma.experiments.registry import EXPERIMENTS


def test_project_config_is_valid(config):
    assert [a.id for a in config.agents.enabled] == ["mathematician", "enthusiast"]
    assert config.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert config.uncertainty.semantic_entropy.n == 10
    assert config.uncertainty.semantic_entropy.top_k is None
    assert config.uncertainty.enabled_metrics() == ["mcsp", "se", "sd", "length"]
    assert len(config.features.order) == 8
    assert config.model.scoring_backend == "same"
    agents = config.agents
    assert (agents.workflow, agents.num_runs, agents.max_rounds, agents.completion_token) == (
        "feedback",
        10,
        10,
        "TASK_DONE",
    )
    mc = config.uncertainty.monte_carlo
    assert mc.num_generations == 10 and mc.length_normalize


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


def test_cache_fingerprints_split_generation_and_features(config):
    """Uncertainty is measured while generating, so its settings key the generation cache; aggregation and
    feature settings only key the feature step."""
    uq = config.uncertainty
    for section, fields in (
        ("monte_carlo", {"length_normalize": False}),
        ("semantic_density", {"nli_model_name": "other/nli"}),
        ("semantic_entropy", {"nli_model_name": "other/nli"}),
    ):
        other = dataclasses.replace(
            config,
            uncertainty=dataclasses.replace(uq, **{section: dataclasses.replace(getattr(uq, section), **fields)}),
        )
        assert other.generation_fingerprint() != config.generation_fingerprint()
    more_runs = dataclasses.replace(config, agents=dataclasses.replace(config.agents, num_runs=3))
    assert more_runs.generation_fingerprint() != config.generation_fingerprint()
    rms = dataclasses.replace(config, aggregation=dataclasses.replace(config.aggregation, token_length="rms"))
    assert rms.generation_fingerprint() == config.generation_fingerprint()
    assert rms.scoring_fingerprint() != config.scoring_fingerprint()


def test_semantic_entropy_must_match_the_shared_samples(raw_config):
    raw_config["uncertainty"]["monte_carlo"]["num_generations"] = 5
    with pytest.raises(ValueError, match="semantic_entropy.n must equal uncertainty.monte_carlo.num_generations"):
        Settings.from_dict(raw_config)
    raw_config["uncertainty"]["semantic_entropy"]["n"] = 5
    assert Settings.from_dict(raw_config).uncertainty.monte_carlo.num_generations == 5


def test_feedback_workflow_settings_are_validated(raw_config):
    for key, value, match in (
        ("max_rounds", 0, "max_rounds"),
        ("num_runs", 0, "num_runs"),
        ("completion_token", " ", "completion_token"),
    ):
        bad = {**raw_config, "agents": {**raw_config["agents"], key: value}}
        with pytest.raises(ValueError, match=match):
            Settings.from_dict(bad)
    raw_config["agents"]["items"][1]["enabled"] = False
    with pytest.raises(ValueError, match="exactly 2"):
        Settings.from_dict(raw_config)


def test_removed_uncertainty_sections_rejected(raw_config):
    for removed in ("predictive_entropy", "nll"):
        raw_config["uncertainty"][removed] = {"enabled": True}
        with pytest.raises(ValueError, match=removed):
            Settings.from_dict(raw_config)
        del raw_config["uncertainty"][removed]


def test_dynaconf_nested_env_override_is_typed(monkeypatch):
    monkeypatch.setenv("SPUMA_MODEL__GENERATION__TEMPERATURE", "0.2")
    monkeypatch.setenv("SPUMA_DATASET__TEST_SAMPLES", "50")
    s = load_settings()
    assert s.model.generation.temperature == 0.2 and s.dataset.test_samples == 50


def test_get_settings_is_cached_until_cleared(monkeypatch):
    first = get_settings()
    assert get_settings() is first
    monkeypatch.setenv("SPUMA_DATASET__TEST_SAMPLES", "3")
    assert get_settings().dataset.test_samples == first.dataset.test_samples
    get_settings.cache_clear()
    assert get_settings().dataset.test_samples == 3


def test_settings_file_from_env(monkeypatch, tmp_path, raw_config):
    raw_config["experiment"]["name"] = "other_experiment"
    path = tmp_path / "other.yaml"
    path.write_text(yaml.safe_dump(raw_config))
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(path))
    assert load_settings().experiment.name == "other_experiment"


def test_unknown_and_missing_keys_name_the_path(raw_config):
    raw_config["dataset"]["imaeg"] = "x"
    with pytest.raises(ValueError, match="dataset.imaeg"):
        Settings.from_dict(raw_config)
    del raw_config["dataset"]["imaeg"]
    del raw_config["agents"]["items"][0]["system_prompt"]
    with pytest.raises(ValueError, match=r"agents\.items\.0\.system_prompt"):
        Settings.from_dict(raw_config)


def test_removed_tool_settings_rejected(raw_config):
    raw_config["agents"]["items"][0]["allowed_tools"] = ["sympy_solve"]
    with pytest.raises(ValueError, match="allowed_tools"):
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
    assert snap["dataset"]["test_samples"] == 500
    json.dumps(snap)


def test_dotenv_exported_not_in_config(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_secret\n")
    config = load_settings(dotenv_path=env)

    assert os.environ["HF_TOKEN"] == "hf_secret"
    assert "hf_secret" not in json.dumps(config.sanitized())


def test_every_yaml_file_is_an_experiments_settings_file():
    found = {p for p in Path(PROJECT_ROOT).rglob("*.y*ml") if ".venv" not in p.parts and "artifacts" not in p.parts}
    assert found == {e.config_file for e in EXPERIMENTS.values()}, found
    for path in found:
        assert yaml.safe_load(path.read_text())
