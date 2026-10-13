"""Typed settings: config/config-qwen-math.yaml, overridable by SPUMA_* env vars.

The project-root `.env` holds secrets only (HF_TOKEN); it is exported into os.environ, where
huggingface_hub picks the token up. It is not a settings field.

Dynaconf only loads and merges the sources; `Settings` embeds one frozen config dataclass per
package (`<package>/config.py`) and is the single object the rest of the code reads, through the
cached `get_settings()`. Each section validates itself in `__post_init__`; `Settings.__post_init__`
checks the rules that span sections (tool permissions, feature order, model references).
"""

import dataclasses
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from dynaconf import Dynaconf

from spuma.agents.config import AgentConfig, AgentsConfig, OrchestrationConfig
from spuma.common.config import LoggingConfig, build_config, to_plain
from spuma.common.utils import stable_hash
from spuma.dataset.config import DatasetConfig
from spuma.evaluation.config import EvaluationConfig
from spuma.experiments.config import ExperimentConfig, ProjectConfig
from spuma.models.config import GenerationConfig, ModelConfig
from spuma.spn.config import SPNConfig
from spuma.tools.config import ToolsConfig
from spuma.uncertainty.config import SAMPLING_FIELDS, AggregationConfig, FeaturesConfig, UncertaintyConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "config-qwen-math.yaml"
DEFAULT_DOTENV_FILE = PROJECT_ROOT / ".env"
ENV_PREFIX = "SPUMA"
SETTINGS_FILE_ENV = "SPUMA_CONFIG_FILE"  # settings file get_settings() loads (CLI --config sets it)
SECRET_PATTERN = re.compile(r"token|secret|password|api_?key|credential", re.IGNORECASE)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    agents: AgentsConfig
    features: FeaturesConfig
    project: ProjectConfig = field(default_factory=ProjectConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    models: dict[str, ModelConfig] = field(default_factory=dict)  # extra named models; `default` is `model`
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    spn: SPNConfig = field(default_factory=SPNConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        agents = self.agents.enabled
        if not agents:
            raise ValueError("agents.items: at least one agent must be enabled")
        ids = [a.id for a in self.agents.items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"agent ids must be unique: {ids}")
        if "default" in self.models:
            raise ValueError("models: 'default' is reserved for the top-level `model` section")
        for a in agents:
            if a.model != "default" and a.model not in self.models:
                raise ValueError(f"agent {a.id!r}: model {a.model!r} is not 'default' or a key of `models`")
            self._check_tools(a.id, a.allowed_tools)
        for name, variant in self.experiment.variants.items():
            for agent_id, tools in variant.allowed_tools.items():
                if agent_id not in ids:
                    raise ValueError(f"experiment.variants.{name}: unknown agent {agent_id!r}")
                self._check_tools(f"{name}/{agent_id}", tools)
        expected = {f"{a.id}_{m}" for a in agents for m in self.uncertainty.enabled_metrics()}
        order = self.features.order
        if len(order) != len(set(order)) or set(order) != expected:
            raise ValueError(
                f"features.order must list each enabled <agent>_<metric> exactly once; "
                f"expected {sorted(expected)}, got {order}"
            )

    def _check_tools(self, owner: str, tools: list[str]) -> None:
        unknown = [t for t in tools if t not in self.tools.registry]
        if unknown:
            raise ValueError(f"{owner}: allowed_tools {unknown} are not registered under tools.registry")
        # Disabled tools may stay listed (runtime disabling); the executor refuses them.

    # --- (de)serialization -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Settings:
        return build_config(cls, raw)

    def to_dict(self) -> dict[str, Any]:
        return to_plain(self)

    def sanitized(self) -> dict[str, Any]:
        """JSON-able snapshot with anything secret-looking redacted (saved with every run)."""
        return _redact(self.to_dict())

    # --- per-agent lookups -----------------------------------------------------------------------

    def all_models(self) -> dict[str, ModelConfig]:
        return {"default": self.model, **self.models}

    def model_for(self, agent: AgentConfig) -> ModelConfig:
        return self.model if agent.model == "default" else self.models[agent.model]

    def generation_for(self, agent: AgentConfig) -> GenerationConfig:
        return agent.generation or self.model_for(agent).generation

    def max_steps_for(self, agent: AgentConfig) -> int:
        return agent.max_steps or self.orchestration.max_steps_per_agent

    def max_tool_calls_for(self, agent: AgentConfig) -> int:
        return agent.max_tool_calls if agent.max_tool_calls is not None else self.orchestration.max_tool_calls_per_agent

    def with_variant(self, name: str) -> Settings:
        """These settings with one experiment variant's `allowed_tools` overrides applied."""
        overrides = self.experiment.variants[name].allowed_tools
        items = [dataclasses.replace(a, allowed_tools=overrides.get(a.id, a.allowed_tools)) for a in self.agents.items]
        return dataclasses.replace(self, agents=dataclasses.replace(self.agents, items=items))

    # --- cache keys ------------------------------------------------------------------------------

    def fingerprint(self, *sections: str) -> str:
        data = self.to_dict()
        return stable_hash({s: data[s] for s in sections} if sections else data)

    def generation_fingerprint(self) -> str:
        """Cache key of the generate phase: trajectories, generation-time scores and semantic-entropy samples."""
        data = {k: v for k, v in self.to_dict().items() if k in ("project", "agents", "orchestration", "tools")}
        uq = self.uncertainty
        data["models"] = {k: m.generation_fingerprint() for k, m in self.all_models().items()}
        # only the sampling side of SE: changing the NLI model re-clusters stored samples, it does not resample
        se = to_plain(uq.semantic_entropy)
        data["semantic_entropy"] = {k: se[k] for k in SAMPLING_FIELDS}
        data["scored_metrics"] = [uq.predictive_entropy.enabled, uq.nll.enabled]
        return stable_hash(data)

    def scoring_fingerprint(self) -> str:
        """Cache key of the score phase: scoring engines + UQ, aggregation and feature settings."""
        data = {k: v for k, v in self.to_dict().items() if k in ("uncertainty", "aggregation", "features")}
        data["scorers"] = {k: m.fingerprint(m.scoring_kind) for k, m in self.all_models().items()}
        return stable_hash(data)


def load_settings(
    settings_files: list[str | Path] | None = None, dotenv_path: str | Path | None = None, **kwargs: object
) -> Settings:
    """Export `.env` (default: project root), then read YAML + SPUMA_* env vars and validate them into
    `Settings`. Extra kwargs go to Dynaconf.

    `.env` is loaded here rather than by Dynaconf because Dynaconf ignores `dotenv_path` at setup and
    searches upward from the CWD instead. Variables already set in the environment win over `.env`.
    Nested env overrides use double underscores: `SPUMA_MODEL__BACKEND=vllm`, `SPUMA_DATASET__TEST_SAMPLES=50`.
    """
    dotenv_file = Path(dotenv_path or DEFAULT_DOTENV_FILE)
    loaded = load_dotenv(dotenv_file, override=False)
    logger.debug(".env %s: %s (values are never logged)", dotenv_file, "exported" if loaded else "not found/empty")
    files = [str(p) for p in settings_files or [os.environ.get(SETTINGS_FILE_ENV) or DEFAULT_SETTINGS_FILE]]
    logger.debug("reading settings files %s + %s_* environment overrides", files, ENV_PREFIX)
    raw = Dynaconf(envvar_prefix=ENV_PREFIX, settings_files=files, load_dotenv=False, merge_enabled=True, **kwargs)
    names = [f.name for f in dataclasses.fields(Settings)]
    sections = {name: raw.get(name) for name in names if raw.get(name) is not None}
    settings = Settings.from_dict(_plain(sections))
    logger.info(
        "settings loaded from %s: experiment=%s model=%s backend=%s/%s agents=%s",
        files,
        settings.experiment.name,
        settings.model.name,
        settings.model.backend,
        settings.model.scoring_kind,
        [(a.id, a.allowed_tools) for a in settings.agents.enabled],
    )
    return settings


@lru_cache(maxsize=32)  # base settings + one entry per experiment variant
def get_settings(variant: str | None = None) -> Settings:
    """Process-wide settings from the default files, loaded once. Every package reads its section from here
    instead of receiving a config argument. `variant` names an `experiment.variants` entry: the same settings
    with that variant's `allowed_tools` applied (also cached). `get_settings.cache_clear()` reloads."""
    if variant is not None:
        return get_settings().with_variant(variant)
    return load_settings()


def _plain(value: object) -> object:
    """Dynaconf boxes -> plain dicts/lists."""
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _redact(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if key and SECRET_PATTERN.search(key) and not key.startswith("max_"):
        return "***"
    return value
