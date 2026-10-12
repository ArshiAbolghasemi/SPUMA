"""Load the single `config/config-qwen-math.yaml` into validated `Settings`.

Sources, lowest to highest precedence:
  1. `config/config-qwen-math.yaml` (the only YAML file in the project),
  2. `SPUMA_<SECTION>__<KEY>[__<KEY>...]` environment variables (values parsed as YAML scalars),
     e.g. `SPUMA_MODEL__BACKEND=vllm`, `SPUMA_DATASET__TEST_SAMPLES=50`.

The project-root `.env` holds secrets only (HF_TOKEN). It is exported to `os.environ`, where
huggingface_hub reads it; secrets are never config fields and never logged.
"""

import logging
import os
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from spuma.config.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "config-qwen-math.yaml"
DEFAULT_DOTENV_FILE = PROJECT_ROOT / ".env"
ENV_PREFIX = "SPUMA_"

logger = logging.getLogger(__name__)


def load_settings(path: str | Path | None = None, dotenv_path: str | Path | None = None) -> Settings:
    """Export `.env` (secrets), read the config file, apply SPUMA_* overrides, validate."""
    dotenv_file = Path(dotenv_path or DEFAULT_DOTENV_FILE)
    loaded = load_dotenv(dotenv_file, override=False)
    logger.debug(".env %s: %s (values are never logged)", dotenv_file, "exported" if loaded else "not found/empty")
    config_file = Path(path or DEFAULT_CONFIG_FILE)
    raw = yaml.safe_load(config_file.read_text()) or {}
    overrides = apply_env_overrides(raw, os.environ)
    if overrides:
        logger.info("config overrides from environment: %s", overrides)
    config = Settings.from_dict(raw)
    logger.info(
        "config loaded from %s: experiment=%s model=%s agents=%s",
        config_file,
        config.experiment.name,
        config.model.name,
        [(a.id, a.allowed_tools) for a in config.agents.enabled],
    )
    return config


def apply_env_overrides(raw: dict[str, Any], environ: Mapping[str, str]) -> list[str]:
    """Set `SPUMA_A__B__C=value` into raw[a][b][c] (value parsed as a YAML scalar). Returns the keys set."""
    applied = []
    for key, value in sorted(environ.items()):
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        path = key[len(ENV_PREFIX) :].lower().split("__")
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"{key}: {part!r} is not a mapping in the config file")
        node[path[-1]] = yaml.safe_load(value)
        applied.append(".".join(path))
    return applied


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings from the default files, loaded once."""
    return load_settings()
