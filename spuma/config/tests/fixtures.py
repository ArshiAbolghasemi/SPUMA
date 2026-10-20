"""Settings fixtures for every test (registered in pyproject: `-p spuma.config.tests.fixtures`)."""

import dataclasses
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from spuma.config import Settings, get_settings
from spuma.config import settings as settings_module


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Never read the project .env or inherit SPUMA_* / HF_TOKEN from the shell."""
    monkeypatch.setattr(settings_module, "DEFAULT_DOTENV_FILE", tmp_path / "no.env")
    for var in {*(v for v in os.environ if v.startswith("SPUMA_")), settings_module.SETTINGS_FILE_ENV, "HF_TOKEN"}:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def config() -> Settings:
    """The process-wide settings (`get_settings()`): the real config/math_ma_qwen_feedbackloop.yaml."""
    return get_settings()


@pytest.fixture
def set_settings(monkeypatch):
    """Make `get_settings()` return the given settings for this test (code reads settings, never receives them)."""

    def install(settings: Settings) -> Settings:
        monkeypatch.setattr(settings_module, "load_settings", lambda *args, **kwargs: settings)
        get_settings.cache_clear()
        return settings

    return install


@pytest.fixture
def raw_config() -> dict[str, Any]:

    return yaml.safe_load(Path(settings_module.DEFAULT_SETTINGS_FILE).read_text())


METRIC_SECTIONS = {
    "mcsp": "monte_carlo_sequence_probability",
    "se": "semantic_entropy",
    "sd": "semantic_density",
    "nll": "nll",
    "length": "token_length",
}


@pytest.fixture
def with_metrics(config, set_settings):
    """Install settings with exactly these metrics enabled (and `features.order` to match)."""

    def install(metrics: list[str]) -> Settings:
        uq = config.uncertainty
        flags = {
            section: dataclasses.replace(getattr(uq, section), enabled=m in metrics)
            for m, section in METRIC_SECTIONS.items()
        }
        order = [f"{a.id}_{m}" for a in config.agents.enabled for m in metrics]
        return set_settings(
            dataclasses.replace(
                config,
                uncertainty=dataclasses.replace(uq, **flags),
                features=dataclasses.replace(config.features, order=order),
            )
        )

    return install
