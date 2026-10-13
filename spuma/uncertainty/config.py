"""`uncertainty:`, `aggregation:` and `features:` sections."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

METRICS = ("pe", "se", "nll", "length")
# Semantic-entropy fields that change the drawn samples (the rest only changes how they are clustered).
SAMPLING_FIELDS = ("enabled", "n", "temperature", "top_p", "top_k", "max_new_tokens", "tool_call_policy")


@dataclass(frozen=True)
class PredictiveEntropyConfig:
    enabled: bool = True
    full_vocabulary: bool = True
    distribution: str = "raw_model"  # untempered, untruncated p_theta(.|context)

    def __post_init__(self) -> None:
        if self.full_vocabulary is not True:
            raise ValueError("uncertainty.predictive_entropy.full_vocabulary must be true (no top-k approximation)")
        check_choice("uncertainty.predictive_entropy.distribution", self.distribution, ("raw_model",))


@dataclass(frozen=True)
class SemanticEntropyConfig:
    """Semantic entropy via UQLM (`uqlm.BlackBoxUQ`, scorer `semantic_negentropy`).

    Sampling (n continuations of each step's frozen context) is done by SPUMA's generation backend; UQLM
    clusters the step's own text together with those samples using an NLI model.
    """

    enabled: bool = True
    backend: str = "uqlm"
    n: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: None = None  # top-k truncation is always disabled for semantic sampling
    max_new_tokens: int | None = None  # None: the agent's max_new_tokens
    # Sampled tool calls are never executed; their text is clustered like any other continuation.
    tool_call_policy: str = "no_execution"
    nli_model_name: str = "microsoft/deberta-large-mnli"
    nli_max_length: int = 2000  # characters kept per text (UQLM truncates longer ones, with a warning)
    nli_batch_size: int = 32
    nli_device: str | None = None  # None: UQLM's choice (CUDA, then MPS, then CPU)

    def __post_init__(self) -> None:
        key = "uncertainty.semantic_entropy"
        check_choice(f"{key}.backend", self.backend, ("uqlm",))
        check_range(f"{key}.n", self.n, ge=2)
        check_range(f"{key}.temperature", self.temperature, gt=0.0)
        check_range(f"{key}.top_p", self.top_p, gt=0.0, le=1.0)
        if self.top_k is not None:
            raise ValueError(f"{key}.top_k must be null: top-k truncation is disabled for semantic sampling")
        check_range(f"{key}.max_new_tokens", self.max_new_tokens, gt=0)
        check_choice(f"{key}.tool_call_policy", self.tool_call_policy, ("no_execution",))
        check_range(f"{key}.nli_max_length", self.nli_max_length, gt=0)
        check_range(f"{key}.nli_batch_size", self.nli_batch_size, gt=0)


@dataclass(frozen=True)
class NLLConfig:
    enabled: bool = True
    normalize_by_length: bool = True


@dataclass(frozen=True)
class TokenLengthConfig:
    enabled: bool = True


@dataclass(frozen=True)
class UncertaintyConfig:
    """`uncertainty:` section."""

    predictive_entropy: PredictiveEntropyConfig = field(default_factory=PredictiveEntropyConfig)
    semantic_entropy: SemanticEntropyConfig = field(default_factory=SemanticEntropyConfig)
    nll: NLLConfig = field(default_factory=NLLConfig)
    token_length: TokenLengthConfig = field(default_factory=TokenLengthConfig)

    def enabled_metrics(self) -> list[str]:
        flags = {
            "pe": self.predictive_entropy.enabled,
            "se": self.semantic_entropy.enabled,
            "nll": self.nll.enabled,
            "length": self.token_length.enabled,
        }
        return [m for m in METRICS if flags[m]]


@dataclass(frozen=True)
class AggregationConfig:
    """`aggregation:` section: per-metric strategy over an agent's steps."""

    predictive_entropy: str = "rms"
    semantic_entropy: str = "rms"
    nll: str = "rms"
    token_length: str = "mean"

    def for_metric(self, metric: str) -> str:
        return {
            "pe": self.predictive_entropy,
            "se": self.semantic_entropy,
            "nll": self.nll,
            "length": self.token_length,
        }[metric]


@dataclass(frozen=True)
class FeaturesConfig:
    """`features:` section: the stable order of the aggregated <agent>_<metric> features."""

    order: list[str]
