"""`tools:` section: the global tool registry and execution settings."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range


@dataclass(frozen=True)
class ToolConfig:
    """One `tools.registry` entry."""

    provider: str
    enabled: bool = True
    timeout_seconds: float = 10.0
    operation: str | None = None  # math providers: operation to expose under this name (default: the name)

    def __post_init__(self) -> None:
        check_range("tools.registry.*.timeout_seconds", self.timeout_seconds, gt=0)


@dataclass(frozen=True)
class ToolExecutionConfig:
    """`tools.execution:` section."""

    log_calls: bool = True
    save_results: bool = True
    max_output_tokens: int = 1024
    on_error: str = "return_error"

    def __post_init__(self) -> None:
        check_range("tools.execution.max_output_tokens", self.max_output_tokens, gt=0)
        check_choice("tools.execution.on_error", self.on_error, ("return_error", "raise"))


@dataclass(frozen=True)
class ToolsConfig:
    """`tools:` section."""

    enabled: bool = True
    registry: dict[str, ToolConfig] = field(default_factory=dict)
    execution: ToolExecutionConfig = field(default_factory=ToolExecutionConfig)
