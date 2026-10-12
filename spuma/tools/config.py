"""`tools:` section: the global tool registry and execution settings."""

from typing import Any, Literal

from pydantic import ConfigDict, Field

from spuma.common.config import StrictConfig


class ToolConfig(StrictConfig):
    model_config = ConfigDict(extra="allow", frozen=True)  # provider-specific options

    enabled: bool = True
    provider: str
    timeout_seconds: float = Field(10.0, gt=0)

    @property
    def options(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class ToolExecutionConfig(StrictConfig):
    log_calls: bool = True
    save_results: bool = True
    max_output_tokens: int = Field(1024, gt=0)
    on_error: Literal["return_error", "raise"] = "return_error"


class ToolsConfig(StrictConfig):
    """`tools:` section."""

    enabled: bool = True
    registry: dict[str, ToolConfig] = Field(default_factory=dict)
    execution: ToolExecutionConfig = Field(default_factory=ToolExecutionConfig)
