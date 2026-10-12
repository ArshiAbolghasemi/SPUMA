"""Settings package: `settings.Settings` (every package's config section in one frozen dataclass) and
`loader` (config/config-qwen-math.yaml + .env + SPUMA_* overrides). Section classes live in each package's `config.py`.

Import from here: `from spuma.config import Settings, load_settings`.
"""

from spuma.config.loader import (
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
    "get_settings",
    "load_settings",
]
