"""Settings package: `settings.Settings` (every package's config section in one frozen dataclass) and
`loader` (config/config-qwen-math.yaml + .env + SPUMA_* overrides). Section classes live in each package's `config.py`.

`get_settings()` (lru_cache) loads the settings once; code reads `get_settings().<section>` instead of receiving
config/settings arguments. `get_settings(variant)` applies an experiment variant's allowed_tools.
"""

from spuma.config.loader import (
    CONFIG_FILE_ENV,
    DEFAULT_CONFIG_FILE,
    DEFAULT_DOTENV_FILE,
    ENV_PREFIX,
    PROJECT_ROOT,
    apply_env_overrides,
    get_settings,
    load_settings,
)
from spuma.config.settings import Settings

__all__ = [
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_DOTENV_FILE",
    "ENV_PREFIX",
    "PROJECT_ROOT",
    "Settings",
    "apply_env_overrides",
    "CONFIG_FILE_ENV",
    "get_settings",
    "load_settings",
]
