"""Step-level uncertainty: PE, NLL, token length, semantic entropy — independent of the inference backend.

Definitions (natural log, raw model distribution p_theta, generated tokens only):
    PE_s  = (1/L) Σ_t -Σ_v p(v|c_t) log p(v|c_t)
    NLL_s = -(1/L) Σ_t log p(y_t|c_t)
    SE_s  = -Σ_c (|c|/n) log(|c|/n) over semantic clusters of n samples from the frozen pre-step context

Where the inputs come from:
  - token scores: stored on the step by the generation backend (`scoring_backend: same`) or filled by
    `TokenScorer`, which teacher-forces the saved prompt/continuation ids on the scoring backend;
  - semantic samples: drawn by `SemanticSampler` with the generation backend (generate phase) and
    stored on the step, so SE can be recomputed or compared across clustering methods offline.
Missing inputs raise `MissingScoresError`; nothing is approximated.
"""

import logging
import math
from collections.abc import Callable
from typing import Any, Protocol

from spuma.agents.trajectory import StepRecord, Trajectory
from spuma.common.config import to_plain
from spuma.common.utils import derive_seed, stable_hash
from spuma.config import get_settings
from spuma.models.backend import BackendCapabilityError, GenerationRequest, ModelBackend, require, scorer_id
from spuma.models.config import GenerationConfig
from spuma.uncertainty.config import SemanticEntropyConfig

logger = logging.getLogger(__name__)

SE_SAMPLES_KEY = "se_samples"


class MissingScoresError(RuntimeError):
    pass


class UncertaintyEstimator(Protocol):
    name: str  # metric key: pe | se | nll | length

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        """(value, details to persist)."""
        ...


def _require_tokens(step: StepRecord) -> None:
    if not step.token_ids:
        raise MissingScoresError(f"{step.problem_id}/{step.agent_id}/{step.step_id}: step generated no tokens")


class PredictiveEntropy:
    """Mean over generated tokens of H_t = -Σ_v p(v|c_t) log p(v|c_t) (full vocabulary, raw model, nats)."""

    name = "pe"

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        _require_tokens(step)
        if step.token_entropies is None or len(step.token_entropies) != len(step.token_ids):
            raise MissingScoresError(f"{step.agent_id}/{step.step_id}: no full-vocabulary entropies recorded")
        return sum(step.token_entropies) / len(step.token_entropies), {"scored_by": step.scored_by}


class NegativeLogLikelihood:
    """-(1/L) Σ_t log p(y_t | y_<t, c) (or the unnormalized sum), raw model distribution."""

    name = "nll"

    def __init__(self) -> None:
        self.normalize = get_settings().uncertainty.nll.normalize_by_length

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        _require_tokens(step)
        if step.token_logprobs is None or len(step.token_logprobs) != len(step.token_ids):
            raise MissingScoresError(f"{step.agent_id}/{step.step_id}: no token log-probabilities recorded")
        total = -sum(step.token_logprobs)
        return (total / len(step.token_logprobs) if self.normalize else total), {"scored_by": step.scored_by}


class TokenLength:
    """Number of model-generated tokens in the step (shared tokenizer; tool output excluded)."""

    name = "length"

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        _require_tokens(step)
        return float(len(step.token_ids)), {}


# --- semantic entropy --------------------------------------------------------------------------------


def _se() -> SemanticEntropyConfig:
    return get_settings().uncertainty.semantic_entropy


def semantic_params(step: StepRecord) -> GenerationConfig:
    config = _se()
    return GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=None,  # no top-k truncation (each backend translates "off")
        max_new_tokens=config.max_new_tokens or step.generation_params["max_new_tokens"],
    )


def semantic_policy() -> dict[str, Any]:
    config = _se()
    return {
        "n": config.n,
        "context": "frozen pre-step prompt_ids incl. prior tool observations",
        "tool_call_policy": config.tool_call_policy,
        "top_k": None,
        "clustering": "uqlm.BlackBoxUQ semantic_negentropy (NLI entailment in either direction)",
        "clustered_texts": "the step's own text + its n samples",
        "nli_model": config.nli_model_name,
        "nli_prompt": None,  # prompts_in_nli=False: step contexts are long chat transcripts
        "probabilities": "discrete (cluster frequency)",
    }


def _sample_key(step: StepRecord) -> str:
    return stable_hash({"params": to_plain(semantic_params(step)), "n": _se().n, "prompt": step.prompt_ids})


class SemanticSampler:
    """Draws the n alternatives of every step from its exact `prompt_ids`, one batched backend call per trajectory.

    Sampled tool calls are never executed (`tool_call_policy: no_execution`) and the main trajectory is
    only read, so sampling cannot change tool state or the recorded steps.
    """

    def __init__(self, backend: ModelBackend) -> None:
        require(backend, "multiple_completions", "batch_generation")
        self.backend, self.config, self.seed = backend, _se(), get_settings().project.seed

    def needs_samples(self, step: StepRecord) -> bool:
        stored = step.uq_details.get(SE_SAMPLES_KEY)
        return not stored or stored.get("key") != _sample_key(step)

    def sample_steps(self, steps: list[StepRecord]) -> None:
        """Fill `uq_details[se_samples]` of the given steps in place."""
        todo = [s for s in steps if self.needs_samples(s)]
        if not todo:
            return
        requests = [
            GenerationRequest(
                prompt_ids=s.prompt_ids,
                params=semantic_params(s),
                seed=derive_seed(self.seed, s.problem_id, s.agent_id, s.step_id, "semantic_entropy"),
                n=self.config.n,
            )
            for s in todo
        ]
        for step, request, results in zip(todo, requests, self.backend.generate_batch(requests), strict=True):
            if len(results) != self.config.n:
                raise MissingScoresError(f"backend returned {len(results)} samples, expected {self.config.n}")
            step.uq_details[SE_SAMPLES_KEY] = {
                "key": _sample_key(step),
                "texts": [r.text for r in results],
                "num_tokens": [r.num_tokens for r in results],
                "finish_reasons": [r.finish_reason for r in results],
                "seed": request.seed,
                "params": to_plain(request.params),
                "backend": self.backend.name,
                "latency_s": sum(r.latency_s for r in results),
                "policy": semantic_policy(),
            }
        logger.info(
            "semantic samples",
            extra={"steps": len(todo), "n": self.config.n, "backend": self.backend.name},
        )


def load_black_box_uq() -> Any:
    """UQLM `BlackBoxUQ` with the `semantic_negentropy` scorer (loads the NLI model)."""
    from uqlm import BlackBoxUQ

    config = _se()
    return BlackBoxUQ(
        llm=None,  # responses come from SPUMA's backends; UQLM only scores them
        scorers=["semantic_negentropy"],
        nli_model_name=config.nli_model_name,
        use_best=False,
        device=config.nli_device,
        max_length=config.nli_max_length,
        nli_batch_size=config.nli_batch_size,
    )


class SemanticEntropy:
    """Semantic entropy of a step from UQLM: `BlackBoxUQ(scorers=["semantic_negentropy"]).score(...)`.

    UQLM clusters the step's own generated text together with its n stored samples with its NLI model (a
    text joins every cluster whose first member it entails or is entailed by) and returns
        semantic_negentropy = 1 - SE / ln(n + 1),   SE = -Σ_c p_c ln p_c,  p_c = cluster frequency over n + 1 texts.
    The step metric is SE in nats, recovered exactly from that score (same scale as PE and NLL);
    `semantic_negentropy` itself is stored with the step. BlackBoxUQ.score passes no prompt to the NLI
    model. The model is loaded once, on first use, and shared by all agents.
    """

    name = "se"

    def __init__(self, uq: Any = None) -> None:
        self.config = _se()
        self._uq = uq

    @property
    def uq(self) -> Any:
        if self._uq is None:
            self._uq = load_black_box_uq()
            logger.info(
                "uqlm BlackBoxUQ ready", extra={"scorer": "semantic_negentropy", "nli": self.config.nli_model_name}
            )
        return self._uq

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        return self.estimate_steps([step])[0]

    def estimate_steps(self, steps: list[StepRecord]) -> list[tuple[float, dict[str, Any]]]:
        """All given steps in one BlackBoxUQ.score call (UQLM has a fixed per-call overhead)."""
        samples = []
        for step in steps:
            stored = step.uq_details.get(SE_SAMPLES_KEY)
            if not stored or stored.get("key") != _sample_key(step):
                raise MissingScoresError(
                    f"{step.problem_id}/{step.agent_id}/{step.step_id}: no semantic samples for the current "
                    "semantic_entropy settings (they are drawn in the generate phase)"
                )
            samples.append(stored["texts"])
        return self.from_texts_batch([s.text for s in steps], samples)

    def from_texts(self, response: str, samples: list[str]) -> tuple[float, dict[str, Any]]:
        return self.from_texts_batch([response], [samples])[0]

    def from_texts_batch(self, responses: list[str], samples: list[list[str]]) -> list[tuple[float, dict[str, Any]]]:
        if not responses:
            return []
        if any(len(s) != self.config.n for s in samples):
            raise MissingScoresError(f"semantic samples per step must be exactly n={self.config.n}")
        result = self.uq.score(responses=responses, sampled_responses=samples, show_progress_bars=False)
        out = []
        for negentropy in map(float, result.data["semantic_negentropy"]):
            if math.isnan(negentropy):
                raise MissingScoresError("UQLM returned NaN semantic_negentropy")
            value = (1.0 - negentropy) * math.log(self.config.n + 1)
            out.append((value, {"semantic_negentropy": negentropy, "nli_model": self.config.nli_model_name}))
        return out


# --- token scoring -----------------------------------------------------------------------------------


class TokenScorer:
    """Ensures every step carries raw-model scores from the configured scoring engine.

    Steps already scored by that engine (e.g. at generation time with `scoring_backend: same`) are
    left alone and the engine is never loaded. Otherwise the saved prompt/continuation ids are
    teacher-forced on the scoring backend, which is created lazily on first use.
    """

    def __init__(self, scorer: str, backend_factory: Callable[[], ModelBackend], full_vocabulary: bool) -> None:
        self.scorer, self.factory, self.full_vocabulary = scorer, backend_factory, full_vocabulary
        self._backend: ModelBackend | None = None

    def is_scored(self, step: StepRecord) -> bool:
        return (
            step.scored_by == self.scorer
            and step.token_logprobs is not None
            and (not self.full_vocabulary or step.token_entropies is not None)
        )

    def ensure(self, step: StepRecord) -> None:
        if self.is_scored(step):
            return
        if self._backend is None:
            self._backend = self.factory()
            require(
                self._backend, "teacher_forced_scoring", *(["full_vocabulary_scoring"] if self.full_vocabulary else [])
            )
        expected = step.generation_meta.get("compat")
        actual = self._backend.info.compatibility_key()
        if expected is not None and expected != actual:
            raise BackendCapabilityError(f"scoring backend differs from the generating one: {actual} vs {expected}")
        scores = self._backend.score(step.prompt_ids, step.token_ids, self.full_vocabulary)
        step.token_logprobs, step.token_entropies, step.scored_by = (
            scores.token_logprobs,
            scores.token_entropies,
            scores.scorer,
        )
        logger.debug(
            "teacher-forced scores",
            extra={"agent_id": step.agent_id, "step_id": step.step_id, "scorer": scores.scorer, "s": scores.latency_s},
        )


def build_estimators() -> dict[str, list[UncertaintyEstimator]]:
    """Per-agent estimator lists in metric order (no backend needed: inputs are stored on the steps)."""
    config = get_settings()
    uq = config.uncertainty
    semantic = SemanticEntropy()  # one NLI model shared by every agent
    out: dict[str, list[UncertaintyEstimator]] = {}
    for agent in config.agents.enabled:
        ests: list[UncertaintyEstimator] = []
        if uq.predictive_entropy.enabled:
            ests.append(PredictiveEntropy())
        if uq.semantic_entropy.enabled:
            ests.append(semantic)
        if uq.nll.enabled:
            ests.append(NegativeLogLikelihood())
        if uq.token_length.enabled:
            ests.append(TokenLength())
        out[agent.id] = ests
    return out


def build_token_scorers(backend_factory: Callable[..., ModelBackend]) -> dict[str, TokenScorer | None]:
    """Per-agent scorer (None when neither PE nor NLL is enabled); agents sharing a model share one."""
    config = get_settings()
    uq = config.uncertainty
    if not (uq.predictive_entropy.enabled or uq.nll.enabled):
        return {a.id: None for a in config.agents.enabled}
    shared: dict[str, TokenScorer] = {}
    out: dict[str, TokenScorer | None] = {}
    for agent in config.agents.enabled:
        model = config.model_for(agent)
        if agent.model not in shared:
            shared[agent.model] = TokenScorer(
                scorer_id(model, model.scoring_kind),
                lambda m=model: backend_factory(m, m.scoring_kind),
                full_vocabulary=uq.predictive_entropy.enabled,
            )
        out[agent.id] = shared[agent.model]
    return out


def score_trajectory(
    trajectory: Trajectory,
    estimators: dict[str, list[UncertaintyEstimator]],
    scorers: dict[str, TokenScorer | None] | None = None,
) -> Trajectory:
    """Fill token scores (if needed), `metrics` and `uq_details` of every step; the input is untouched."""
    scored = trajectory.model_copy(deep=True)
    # Estimators that score many steps at once (UQLM) get every step of the trajectory in one call.
    batched: dict[int, tuple[Any, list[StepRecord]]] = {}
    for trace in scored.agents:
        for est in estimators[trace.agent_id]:
            if hasattr(est, "estimate_steps"):
                batched.setdefault(id(est), (est, []))[1].extend(trace.steps)
    precomputed: dict[tuple[int, int], tuple[float, dict[str, Any]]] = {}
    for est, steps in batched.values():
        for step, res in zip(steps, est.estimate_steps(steps), strict=True):
            precomputed[(id(est), id(step))] = res
    for trace in scored.agents:
        scorer = (scorers or {}).get(trace.agent_id)
        for step in trace.steps:
            if scorer is not None:
                scorer.ensure(step)
            for est in estimators[trace.agent_id]:
                value, details = precomputed.get((id(est), id(step))) or est.estimate(step)
                step.metrics[est.name] = value
                if details:
                    step.uq_details[est.name] = details
            logger.info(
                "step uncertainty",
                extra={
                    "problem_id": step.problem_id,
                    "agent_id": step.agent_id,
                    "step_id": step.step_id,
                    "metrics": step.metrics,
                    "scored_by": step.scored_by,
                    "tool_calls": [{"tool": r["tool"], "status": r["status"]} for r in step.tool_records],
                },
            )
    return scored


def step_metric_rows(trajectory: Trajectory) -> list[dict[str, Any]]:
    """The per-step persistence format: problem, agent, step, metrics, tool-call summary."""
    return [
        {
            "problem_id": s.problem_id,
            "agent_id": s.agent_id,
            "step_id": s.step_id,
            "metrics": s.metrics,
            "tool_calls": [{"tool": r["tool"], "status": r["status"]} for r in s.tool_records],
        }
        for a in trajectory.agents
        for s in a.steps
    ]
