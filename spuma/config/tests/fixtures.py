"""Settings fixtures for every test (registered in pyproject: `-p spuma.config.tests.fixtures`)."""

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
    # SPUMA_CONFIG_FILE always listed: Experiment.activate() sets it, and teardown must remove it again
    for var in {*(v for v in os.environ if v.startswith("SPUMA_")), settings_module.SETTINGS_FILE_ENV, "HF_TOKEN"}:
        monkeypatch.setenv(var, "")  # record the original so teardown restores it (load_dotenv writes os.environ)
        monkeypatch.delenv(var)
    get_settings.cache_clear()  # every test starts from a fresh load of the real config file
    yield
    get_settings.cache_clear()


@pytest.fixture
def config() -> Settings:
    """The process-wide settings (`get_settings()`): the real config/config-qwen-math.yaml."""
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
