"""`uncertainty:`, `aggregation:` and `features:` sections."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

METRICS = ("mcsp", "se", "sd", "nll", "length")
SAMPLED_METRICS = ("mcsp", "se", "sd")


@dataclass(frozen=True)
class SemanticEntropyConfig:
    """White-box semantic entropy via UQLM's `SemanticEntropy` scorer on the shared Monte Carlo samples.

    The response and its samples are clustered by meaning (`equivalence`); cluster probabilities are the
    normalized sums of the members' sequence probabilities under the raw model. The sampling fields must
    equal `uncertainty.monte_carlo`'s (SE uses those samples).
    """

    enabled: bool = True
    backend: str = "uqlm"
    n: int = 10
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: None = None
    max_new_tokens: int | None = None
    equivalence: str = "answer_aware"
    nli_model_name: str = "tasksource/ModernBERT-large-nli"
    nli_max_length: int = 8000
    nli_max_tokens: int | None = 2048
    nli_batch_size: int = 32
    nli_device: str | None = None

    def __post_init__(self) -> None:
        key = "uncertainty.semantic_entropy"
        check_choice(f"{key}.backend", self.backend, ("uqlm",))
        check_range(f"{key}.n", self.n, ge=2)
        check_range(f"{key}.temperature", self.temperature, gt=0.0)
        check_range(f"{key}.top_p", self.top_p, gt=0.0, le=1.0)
        if self.top_k is not None:
            raise ValueError(f"{key}.top_k must be null: top-k truncation is disabled for semantic sampling")
        check_range(f"{key}.max_new_tokens", self.max_new_tokens, gt=0)
        check_range(f"{key}.nli_max_length", self.nli_max_length, gt=0)
        check_choice(f"{key}.equivalence", self.equivalence, ("answer_aware", "nli"))
        check_range(f"{key}.nli_max_tokens", self.nli_max_tokens, gt=0)
        check_range(f"{key}.nli_batch_size", self.nli_batch_size, gt=0)


@dataclass(frozen=True)
class MonteCarloConfig:
    """`uncertainty.monte_carlo`: the samples drawn from each step's frozen context for MCSP and SD.

    Every sample has its own seed, derived from (project.seed, problem, run, agent, round, sample index).
    `length_normalize` sets the sequence probability the UQLM scorers use: exp(mean_t log p(y_t)) when true,
    prod_t p(y_t) when false (raw model distribution).
    """

    num_generations: int = 10
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    max_new_tokens: int | None = None
    length_normalize: bool = True

    def __post_init__(self) -> None:
        key = "uncertainty.monte_carlo"
        check_range(f"{key}.num_generations", self.num_generations, ge=2)
        check_range(f"{key}.temperature", self.temperature, gt=0.0)
        check_range(f"{key}.top_p", self.top_p, gt=0.0, le=1.0)
        check_range(f"{key}.top_k", self.top_k, gt=0)
        check_range(f"{key}.max_new_tokens", self.max_new_tokens, gt=0)


@dataclass(frozen=True)
class SemanticDensityConfig:
    """UQLM white-box `semantic_density` (Qiu et al., 2024); the NLI model scores response/sample pairs."""

    enabled: bool = False
    nli_model_name: str = "tasksource/ModernBERT-large-nli"
    nli_max_length: int = 8000
    nli_max_tokens: int | None = 2048
    nli_batch_size: int = 32
    nli_device: str | None = None

    def __post_init__(self) -> None:
        key = "uncertainty.semantic_density"
        check_range(f"{key}.nli_max_length", self.nli_max_length, gt=0)
        check_range(f"{key}.nli_max_tokens", self.nli_max_tokens, gt=0)
        check_range(f"{key}.nli_batch_size", self.nli_batch_size, gt=0)


@dataclass(frozen=True)
class MonteCarloSequenceProbabilityConfig:
    """UQLM white-box `monte_carlo_probability`."""

    enabled: bool = True


@dataclass(frozen=True)
class NLLConfig:
    """Token NLL of the response: -(1/L) Σ_t log p(y_t)."""

    enabled: bool = True


@dataclass(frozen=True)
class TokenLengthConfig:
    enabled: bool = True


@dataclass(frozen=True)
class UncertaintyConfig:
    """`uncertainty:` section: one section per per-step metric, each with an `enabled` flag (a disabled metric
    is not computed), plus the shared Monte Carlo sampling."""

    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    monte_carlo_sequence_probability: MonteCarloSequenceProbabilityConfig = field(
        default_factory=MonteCarloSequenceProbabilityConfig
    )
    semantic_entropy: SemanticEntropyConfig = field(default_factory=SemanticEntropyConfig)
    semantic_density: SemanticDensityConfig = field(default_factory=SemanticDensityConfig)
    nll: NLLConfig = field(default_factory=NLLConfig)
    token_length: TokenLengthConfig = field(default_factory=TokenLengthConfig)

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("uncertainty: enable at least one metric")
        se, mc = self.semantic_entropy, self.monte_carlo
        if se.enabled:
            shared = {"n": "num_generations", "temperature": "temperature", "top_p": "top_p", "top_k": "top_k"}
            shared["max_new_tokens"] = "max_new_tokens"
            for se_key, mc_key in shared.items():
                if getattr(se, se_key) != getattr(mc, mc_key):
                    raise ValueError(
                        f"uncertainty.semantic_entropy.{se_key} must equal uncertainty.monte_carlo.{mc_key}: "
                        "SE is computed on the shared Monte Carlo samples"
                    )

    @property
    def metrics(self) -> list[str]:
        """The enabled metrics, in `METRICS` order."""
        flags = {
            "mcsp": self.monte_carlo_sequence_probability.enabled,
            "se": self.semantic_entropy.enabled,
            "sd": self.semantic_density.enabled,
            "nll": self.nll.enabled,
            "length": self.token_length.enabled,
        }
        return [m for m in METRICS if flags[m]]

    @property
    def monte_carlo_enabled(self) -> bool:
        """The shared Monte Carlo samples (with token log-probs) are needed (MCSP, SE or SD enabled)."""
        return any(m in self.metrics for m in SAMPLED_METRICS)

    @property
    def token_logprobs_needed(self) -> bool:
        """Raw token log-probs of the responses are needed (a sampled metric or NLL enabled)."""
        return self.monte_carlo_enabled or "nll" in self.metrics

    def enabled_metrics(self) -> list[str]:
        return self.metrics


@dataclass(frozen=True)
class AggregationConfig:
    """`aggregation:` section: per-metric strategy over an agent's steps."""

    monte_carlo_sequence_probability: str = "rms"
    semantic_entropy: str = "rms"
    semantic_density: str = "rms"
    nll: str = "rms"
    token_length: str = "mean"

    def for_metric(self, metric: str) -> str:
        return {
            "mcsp": self.monte_carlo_sequence_probability,
            "se": self.semantic_entropy,
            "sd": self.semantic_density,
            "nll": self.nll,
            "length": self.token_length,
        }[metric]


@dataclass(frozen=True)
class FeaturesConfig:
    """`features:` section: the stable order of the aggregated <agent>_<metric> features."""

    order: list[str]
