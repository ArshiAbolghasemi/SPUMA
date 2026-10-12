"""Typed settings: one frozen dataclass embedding every package's config section.

Each package owns its section in `<package>/config.py`; `Settings` composes them and is the single
object the rest of the code reads from. It is a pydantic dataclass, so building it from
`config/config-qwen-math.yaml` validates every section and the cross-section rules below (tool permissions,
feature order, model references) before any model is loaded.
"""

import dataclasses
import re
from typing import Any

from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass

from spuma.agents.config import AgentConfig, AgentsConfig, OrchestrationConfig
from spuma.common.config import LoggingConfig
from spuma.common.utils import stable_hash
from spuma.dataset.config import DatasetConfig
from spuma.evaluation.config import EvaluationConfig
from spuma.experiments.config import ExperimentConfig, ProjectConfig
from spuma.models.config import GenerationConfig, ModelConfig
from spuma.spn.config import SPNConfig
from spuma.tools.config import ToolsConfig
from spuma.uncertainty.config import SAMPLING_FIELDS, AggregationConfig, FeaturesConfig, UncertaintyConfig

SECRET_PATTERN = re.compile(r"token|secret|password|api_?key|credential", re.IGNORECASE)


@dataclass(frozen=True, config=ConfigDict(extra="forbid"))
class Settings:
    agents: AgentsConfig
    features: FeaturesConfig
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)  # extra named models; `default` is `model`
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    spn: SPNConfig = Field(default_factory=SPNConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _cross_section(self) -> Settings:
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
        return self

    def _check_tools(self, owner: str, tools: list[str]) -> None:
        unknown = [t for t in tools if t not in self.tools.registry]
        if unknown:
            raise ValueError(f"{owner}: allowed_tools {unknown} are not registered under tools.registry")
        # Disabled tools may stay listed (runtime disabling); the executor refuses them.

    # --- (de)serialization -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Settings:
        return _ADAPTER.validate_python(raw)

    def to_dict(self) -> dict[str, Any]:
        return _ADAPTER.dump_python(self, mode="json")

    def replace(self, **changes: Any) -> Settings:
        """A validated copy with some sections replaced."""
        return dataclasses.replace(self, **changes)

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
        items = [
            a.model_copy(update={"allowed_tools": overrides.get(a.id, a.allowed_tools)}) for a in self.agents.items
        ]
        return self.replace(agents=self.agents.model_copy(update={"items": items}))

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
        data["semantic_entropy"] = uq.semantic_entropy.model_dump(mode="json", include=set(SAMPLING_FIELDS))
        data["scored_metrics"] = [uq.predictive_entropy.enabled, uq.nll.enabled]
        return stable_hash(data)

    def scoring_fingerprint(self) -> str:
        """Cache key of the score phase: scoring engines + UQ, aggregation and feature settings."""
        data = {k: v for k, v in self.to_dict().items() if k in ("uncertainty", "aggregation", "features")}
        data["scorers"] = {k: m.fingerprint(m.scoring_kind) for k, m in self.all_models().items()}
        return stable_hash(data)


_ADAPTER = TypeAdapter(Settings)


def _redact(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    if key and SECRET_PATTERN.search(key) and not key.startswith("max_"):
        return "***"
    return value
