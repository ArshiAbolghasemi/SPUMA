"""`chmm:` section: situational (CHMM) weighting of saved step uncertainties, post-generation."""

from dataclasses import dataclass, field

from spuma.common.config import check_choice, check_range

DISTANCE_ESTIMATORS = ("qa", "lexical")


@dataclass(frozen=True)
class DistanceConfig:
    """`chmm.distance:` the situational-distance estimator (see `spuma.chmm.distances`)."""

    estimator: str = "qa"
    model: str = "deepset/roberta-base-squad2"
    revision: str | None = None
    max_length: int = 384
    stride: int = 128
    max_question_tokens: int = 128
    max_answer_tokens: int = 30
    batch_size: int = 16
    device: str | None = None

    def __post_init__(self) -> None:
        check_choice("chmm.distance.estimator", self.estimator, DISTANCE_ESTIMATORS)
        check_range("chmm.distance.stride", self.stride, gt=0, lt=self.max_length)


@dataclass(frozen=True)
class HMMConfig:
    """`chmm.hmm:` hmmlearn `GaussianHMM` arguments."""

    n_components: int = 3
    covariance_type: str = "diag"
    n_iter: int = 200
    random_state: int = 42

    def __post_init__(self) -> None:
        check_choice("chmm.hmm.n_components", self.n_components, (3,))
        check_choice("chmm.hmm.covariance_type", self.covariance_type, ("diag",))
        check_range("chmm.hmm.n_iter", self.n_iter, ge=1)


@dataclass(frozen=True)
class CHMMConfig:
    """`chmm:` section: CHMM-weighted aggregation, trained in the `train` phase and scored in `evaluate`.

    Post-generation only (saved trajectories and step metrics); see `spuma.chmm.experiment`. An OOD evaluation
    of the experiment reuses the trained CHMM, calibrators and SPNs, frozen.
    """

    enabled: bool = False
    alphas: list[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    primary_alpha: float = 1.0
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    bootstrap_iterations: int = 10000
    confidence: float = 0.95
    # SPN inputs: per_agent (feedback: each agent's responses aggregated; star/sequential: each response on its
    # own) or pooled (all responses aggregated into one [SE, MCSP, NLL, length])
    feature_layout: str = "per_agent"
    # registered experiments (same architecture, e.g. the MATH one) whose trained CHMM, calibrators and SPNs are
    # also evaluated here, frozen: nothing is fitted on this dataset's labels (frozen OOD protocol)
    frozen_from: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "alphas", [float(a) for a in self.alphas])
        object.__setattr__(self, "primary_alpha", float(self.primary_alpha))
        if any(a <= 0 for a in self.alphas):
            raise ValueError("chmm.alphas must be > 0 (alpha = 0, uniform weights, is a unit test, not an experiment)")
        if self.primary_alpha not in self.alphas:
            raise ValueError(f"chmm.primary_alpha {self.primary_alpha} must be one of chmm.alphas {self.alphas}")
        check_choice("chmm.feature_layout", self.feature_layout, ("per_agent", "pooled"))
        check_range("chmm.bootstrap_iterations", self.bootstrap_iterations, gt=0)
        check_range("chmm.confidence", self.confidence, gt=0.0, lt=1.0)
