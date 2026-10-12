"""`uncertainty:`, `aggregation:` and `features:` sections."""

from typing import Literal

from pydantic import Field

from spuma.common.config import StrictConfig

METRICS = ("pe", "se", "nll", "length")
# Semantic-entropy fields that change the drawn samples (the rest only changes how they are clustered).
SAMPLING_FIELDS = ("enabled", "n", "temperature", "top_p", "top_k", "max_new_tokens", "tool_call_policy")


class PredictiveEntropyConfig(StrictConfig):
    enabled: bool = True
    full_vocabulary: Literal[True] = True
    distribution: Literal["raw_model"] = "raw_model"  # untempered, untruncated p_theta(.|context)


class SemanticEntropyConfig(StrictConfig):
    """Semantic entropy via UQLM's NLI clustering (uqlm.scorers.SemanticEntropy, discrete entropy).

    Sampling (n continuations of each step's frozen context) is done by SPUMA's generation backend; UQLM
    clusters the step's own text together with those samples using an NLI model.
    """

    enabled: bool = True
    backend: Literal["uqlm"] = "uqlm"
    n: int = Field(10, ge=2)
    temperature: float = Field(0.7, gt=0.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    top_k: None = None  # top-k truncation is always disabled for semantic sampling
    max_new_tokens: int | None = Field(None, gt=0)  # None: the agent's max_new_tokens
    # Sampled tool calls are never executed; their text is clustered like any other continuation.
    tool_call_policy: Literal["no_execution"] = "no_execution"
    nli_model_name: str = "microsoft/deberta-large-mnli"
    nli_max_length: int = Field(2000, gt=0)  # characters kept per text (UQLM truncates longer ones, with a warning)
    nli_batch_size: int = Field(32, gt=0)
    nli_device: str | None = None  # None: UQLM's choice (CUDA, then MPS, then CPU)


class NLLConfig(StrictConfig):
    enabled: bool = True
    normalize_by_length: bool = True


class TokenLengthConfig(StrictConfig):
    enabled: bool = True


class UncertaintyConfig(StrictConfig):
    """`uncertainty:` section."""

    predictive_entropy: PredictiveEntropyConfig = Field(default_factory=PredictiveEntropyConfig)
    semantic_entropy: SemanticEntropyConfig = Field(default_factory=SemanticEntropyConfig)
    nll: NLLConfig = Field(default_factory=NLLConfig)
    token_length: TokenLengthConfig = Field(default_factory=TokenLengthConfig)

    def enabled_metrics(self) -> list[str]:
        flags = {
            "pe": self.predictive_entropy.enabled,
            "se": self.semantic_entropy.enabled,
            "nll": self.nll.enabled,
            "length": self.token_length.enabled,
        }
        return [m for m in METRICS if flags[m]]


class AggregationConfig(StrictConfig):
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


class FeaturesConfig(StrictConfig):
    """`features:` section: the stable order of the aggregated <agent>_<metric> features."""

    order: list[str]
