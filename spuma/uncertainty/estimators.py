"""Step-level uncertainty: MCSP, SE, SD and token length, measured right after each agent response.

Definitions (natural log, raw model distribution p_theta, generated tokens only; y^0 = the step's own
response, y^1..y^n = its Monte Carlo samples from the same frozen pre-step context c):
    P(y)    = exp((1/L) Σ_t log p(y_t|c, y_<t))   (length_normalize; else Π_t p(y_t|c, y_<t))
    MCSP_s  = (1/(n+1)) Σ_{i=0..n} P(y^i)                          UQLM white-box `monte_carlo_probability`
    NLL_s   = -(1/L) Σ_t log p(y^0_t|c, y^0_<t)                    token NLL of the response
    SE_s    = -Σ_c p_c ln p_c over semantic clusters of the response and its SE samples (SPUMA, unchanged)
    SD_s    = Σ_i P(y^i) K(y^0, y^i) / Σ_i P(y^i),  i = 1..n      UQLM white-box `semantic_density`
              K = (1 - E[d²]) 1[E[d] <= 1],  E[d] = p_c + p_n/√2,  E[d²] = p_c + p_n/2
              (p_c, p_n: NLI contradiction/neutral of "problem⏎y^0" -> "problem⏎y^i")
    length  = generated tokens
MCSP and SD are confidences (higher = more certain); SE and NLL are uncertainties. Only the metrics listed in
`uncertainty.metrics` are computed.

Inputs: token log-probs of the response and of every Monte Carlo sample come from the generating engine
(`scoring_backend: same`, raw distribution). Text removed before scoring (the completion marker) is
teacher-forced on that engine. MCSP, SE and SD all use the same Monte Carlo samples (one seed per sample);
SE clusters them with its own equivalence (`SemanticEntropy`).
Missing inputs raise `MissingScoresError`; nothing is approximated.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from uqlm.nli.nli import NLI
from uqlm.scorers import SemanticDensity as UQLMSemanticDensity
from uqlm.scorers import SemanticEntropy as UQLMSemanticEntropy
from uqlm.white_box.sampled_logprobs import SampledLogprobsScorer

from spuma.agents.trajectory import StepRecord, Trajectory
from spuma.common.config import to_plain
from spuma.common.utils import derive_seed
from spuma.config import get_settings
from spuma.models.backend import GenerationRequest, ModelBackend, require
from spuma.models.config import GenerationConfig
from spuma.uncertainty.config import MonteCarloConfig, SemanticEntropyConfig
from spuma.uncertainty.equivalence import AnswerAwareNLI

logger = logging.getLogger(__name__)

MC_SAMPLES_KEY = "mc_samples"


class MissingScoresError(RuntimeError):
    pass


class UncertaintyEstimator(Protocol):
    name: str

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        """(value, details to persist)."""
        ...


def _where(step: StepRecord) -> str:
    return f"{step.problem_id}/{step.agent_id}/{step.step_id}"


def _require_tokens(step: StepRecord) -> None:
    if not step.token_ids:
        raise MissingScoresError(f"{_where(step)}: step generated no tokens")


class TokenLength:
    """Number of model-generated tokens in the step (shared tokenizer)."""

    name = "length"

    def estimate(self, step: StepRecord) -> tuple[float, dict[str, Any]]:
        _require_tokens(step)
        return float(len(step.token_ids)), {}


def _se() -> SemanticEntropyConfig:
    return get_settings().uncertainty.semantic_entropy


class LabelOrderedNLI(NLI):
    """UQLM's NLI for any 3-way NLI model: outputs reordered to UQLM's fixed [contradiction, neutral, entailment].

    UQLM reads column 0 as contradiction and column 2 as entailment; a model that orders its labels
    differently (tasksource/ModernBERT-large-nli: entailment, neutral, contradiction) would otherwise be
    read backwards. The order comes from the model's own `config.id2label`. `max_tokens` is the token
    budget of a (premise, hypothesis) pair (None: the model's own limit).
    """

    def __init__(self, max_tokens: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        missing = [name for name in self.label_mapping if name not in labels]
        if missing:
            raise ValueError(f"NLI model labels {sorted(labels)} lack {missing}")
        self.order = [labels[name] for name in self.label_mapping]
        if max_tokens is not None:
            self.max_tokens = max_tokens

    def predict_batch(self, pairs: list[tuple[str, str]]) -> Any:
        return super().predict_batch(pairs)[:, self.order]


def load_nli() -> Any:
    config = _se()
    return LabelOrderedNLI(
        nli_model_name=config.nli_model_name,
        max_length=config.nli_max_length,
        max_tokens=config.nli_max_tokens,
        device=config.nli_device,
        batch_size=config.nli_batch_size,
    )


def load_uqlm_semantic_entropy() -> Any:
    """UQLM's semantic-entropy scorer on our equivalence, length-normalized like MCSP.

    `equivalence: answer_aware` (default): math-verify for texts that box an answer, the NLI model for the
    rest (`AnswerAwareNLI`, which loads the model only when needed). `nli`: the NLI model for every pair.
    """
    nli = AnswerAwareNLI(load_nli) if _se().equivalence == "answer_aware" else load_nli()
    return UQLMSemanticEntropy(
        llm=None, use_best=False, prompts_in_nli=False, nli=nli, length_normalize=_mc().length_normalize
    )


class SemanticEntropy:
    """White-box semantic entropy of a step from UQLM's `SemanticEntropy` scorer (`tokenprob` entropy).

    UQLM clusters the step's response with its n Monte Carlo samples (a text joins every cluster whose first
    member is equivalent to it: same math-verified answer, else NLI entailment in either direction) and returns
        SE = -Σ_c p_c ln p_c,  p_c = Σ_{y in c} P(y) / Σ_y P(y),  P(y) = exp(mean_t log p(y_t))   (nats)
    with P from the raw model (Kuhn et al.). `semantic_negentropy` = 1 - SE / ln(n + 1) is stored with it.
    No prompt is passed to the NLI model. The model is loaded once, on first use, and shared by all agents.
    """

    name = "se"

    def __init__(self, uq: Any = None) -> None:
        self.config = _se()
        self._uq = uq

    @property
    def uq(self) -> Any:
        if self._uq is None:
            self._uq = load_uqlm_semantic_entropy()
            logger.info(
                "uqlm semantic entropy ready",
                extra={"nli": self.config.nli_model_name, "max_tokens": self.config.nli_max_tokens},
            )
        return self._uq

    def from_logprobs(
        self, response: str, samples: list[str], response_logprobs: list[float], sample_logprobs: list[list[float]]
    ) -> tuple[float, dict[str, Any]]:
        """Clusters weighted by the texts' sequence probabilities under the raw model."""
        if len(samples) != self.config.n or len(sample_logprobs) != self.config.n:
            raise MissingScoresError(f"semantic samples per step must be exactly n={self.config.n}")
        result = self.uq.score(
            responses=[response],
            sampled_responses=[samples],
            logprobs_results=[_token_dicts(response_logprobs)],
            sampled_logprobs_results=[[_token_dicts(lps) for lps in sample_logprobs]],
            show_progress_bars=False,
        )
        value = _finite(result.data["tokenprob_entropy_values"][0], "semantic entropy")
        return value, {
            "uqlm_scorer": "semantic_entropy (tokenprob)",
            "semantic_negentropy": 1.0 - value / math.log(self.config.n + 1),
            "equivalence": self.config.equivalence,
            "nli_model": self.config.nli_model_name,
        }


def _mc() -> MonteCarloConfig:
    return get_settings().uncertainty.monte_carlo


def monte_carlo_params(step: StepRecord) -> GenerationConfig:
    config = _mc()
    return GenerationConfig(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_new_tokens=config.max_new_tokens or step.generation_params["max_new_tokens"],
    )


def monte_carlo_seeds(step: StepRecord) -> list[int]:
    """One deterministic seed per sample from (project.seed, problem, agent, round, sample index)."""
    base, n = get_settings().project.seed, _mc().num_generations
    return [derive_seed(base, step.problem_id, step.agent_id, step.round, "uqlm", i) for i in range(n)]


class MonteCarloSampler:
    """Draws the Monte Carlo samples of a step from its exact `prompt_ids`, one n=1 request per sample (its own
    seed), all in one batched backend call, with each sample's raw token log-probs. The samples never enter
    the conversation."""

    def __init__(self, backend: ModelBackend) -> None:
        require(backend, "multiple_completions", "batch_generation", "generated_token_logprobs")
        self.backend = backend

    def sample_steps(self, steps: list[StepRecord]) -> None:
        """Fill `uq_details[mc_samples]` of the given steps in place."""
        seeds = {id(s): monte_carlo_seeds(s) for s in steps}
        requests = [
            GenerationRequest(s.prompt_ids, monte_carlo_params(s), seed=seed, n=1, logprobs="token")
            for s in steps
            for seed in seeds[id(s)]
        ]
        outputs = iter(self.backend.generate_batch(requests))
        for step in steps:
            results = [r for _ in seeds[id(step)] for r in next(outputs)]
            if len(results) != len(seeds[id(step)]) or any(r.token_logprobs is None for r in results):
                raise MissingScoresError(f"{_where(step)}: Monte Carlo samples or their log-probabilities missing")
            step.uq_details[MC_SAMPLES_KEY] = {
                "texts": [r.text for r in results],
                "token_ids": [r.token_ids for r in results],
                "token_logprobs": [r.token_logprobs for r in results],
                "scored_by": results[0].scorer,
                "finish_reasons": [r.finish_reason for r in results],
                "seeds": seeds[id(step)],
                "params": to_plain(monte_carlo_params(step)),
                "backend": self.backend.name,
            }


@dataclass
class UQView:
    """What the scorers see for one step: texts and raw token log-probs with the completion marker removed."""

    text: str
    token_logprobs: list[float]
    mc_texts: list[str]
    mc_logprobs: list[list[float]]
    length: int
    rescored: int
    marker_only: int


def _strip(text: str, marker: str | None) -> str | None:
    """`text` without `marker`, or None when there is nothing to strip, or nothing would be left (a response
    that is only the marker is scored as it is: its content is the completion decision)."""
    if not marker or marker not in text:
        return None
    return text.replace(marker, "").strip() or None


def build_view(step: StepRecord, engine: ModelBackend, marker: str | None) -> UQView:
    """The step's response and samples as scored. Texts holding `marker` are stripped and their log-probs
    teacher-forced on `engine` (the generating engine), so the marker never enters a score; a text that is
    only the marker keeps its generated tokens."""
    _require_tokens(step)
    if step.token_logprobs is None or len(step.token_logprobs) != len(step.token_ids):
        raise MissingScoresError(f"{_where(step)}: no token log-probabilities recorded for the response")
    rescored = marker_only = 0

    def scored(text: str, ids: list[int], lps: list[float]) -> tuple[str, list[int], list[float]]:
        nonlocal rescored, marker_only
        stripped = _strip(text, marker)
        if stripped is None:
            marker_only += bool(marker and marker in text)
            return text, ids, lps
        rescored += 1
        new_ids = engine.encode(stripped)
        return stripped, new_ids, engine.score(step.prompt_ids, new_ids, False).token_logprobs

    text, ids, lps = scored(step.text, step.token_ids, step.token_logprobs)
    mc = step.uq_details.get(MC_SAMPLES_KEY) or {"texts": [], "token_ids": [], "token_logprobs": []}
    samples = [scored(*x) for x in zip(mc["texts"], mc["token_ids"], mc["token_logprobs"], strict=True)]
    mc_texts, mc_lps = [s[0] for s in samples], [s[2] for s in samples]
    return UQView(text, lps, mc_texts, mc_lps, len(ids), rescored, marker_only)


def _token_dicts(logprobs: list[float]) -> list[dict[str, float]]:
    return [{"logprob": lp} for lp in logprobs]


def _finite(value: Any, what: str) -> float:
    out = float(value)
    if math.isnan(out):
        raise MissingScoresError(f"UQLM returned NaN {what}")
    return out


class MonteCarloSequenceProbability:
    """UQLM's white-box `monte_carlo_probability` (`SampledLogprobsScorer`): the mean sequence probability of the
    response and its n samples under the raw model. A confidence."""

    name = "mcsp"

    def __init__(self) -> None:
        self.length_normalize = _mc().length_normalize
        self.scorer = SampledLogprobsScorer(scorers=["monte_carlo_probability"], length_normalize=self.length_normalize)

    def score(self, view: UQView, problem: str) -> tuple[float, dict[str, Any]]:
        result = self.scorer.evaluate(
            responses=[view.text],
            sampled_responses=[view.mc_texts],
            logprobs_results=[_token_dicts(view.token_logprobs)],
            sampled_logprobs_results=[[_token_dicts(x) for x in view.mc_logprobs]],
        )
        value = _finite(result["monte_carlo_probability"][0], "Monte Carlo sequence probability")
        return value, {"uqlm_scorer": "monte_carlo_probability", "length_normalize": self.length_normalize}


def load_density_nli() -> Any:
    config = get_settings().uncertainty.semantic_density
    return LabelOrderedNLI(
        nli_model_name=config.nli_model_name,
        max_length=config.nli_max_length,
        max_tokens=config.nli_max_tokens,
        device=config.nli_device,
        batch_size=config.nli_batch_size,
    )


class SemanticDensity:
    """UQLM's white-box `semantic_density` (the `SemanticDensity` scorer `WhiteBoxUQ` runs): NLI kernel between
    the response and each Monte Carlo sample, conditioned on the problem text, weighted by the samples'
    sequence probabilities under the raw model. A confidence. The NLI model loads once, on first use."""

    name = "sd"

    def __init__(self, uq: Any = None) -> None:
        self.config = get_settings().uncertainty.semantic_density
        self.length_normalize = _mc().length_normalize
        self._uq = uq

    @property
    def uq(self) -> Any:
        if self._uq is None:
            self._uq = UQLMSemanticDensity(llm=None, length_normalize=self.length_normalize, nli=load_density_nli())
            logger.info("uqlm semantic density ready", extra={"nli": self.config.nli_model_name})
        return self._uq

    def score(self, view: UQView, problem: str) -> tuple[float, dict[str, Any]]:
        result = self.uq.score(
            prompts=[problem],
            responses=[view.text],
            sampled_responses=[view.mc_texts],
            logprobs_results=[_token_dicts(view.token_logprobs)],
            sampled_logprobs_results=[[_token_dicts(x) for x in view.mc_logprobs]],
            show_progress_bars=False,
        )
        value = _finite(result.data["semantic_density_values"][0], "semantic density")
        details = {"uqlm_scorer": "semantic_density", "nli_model": self.config.nli_model_name, "nli_prompt": "problem"}
        return value, details | {"length_normalize": self.length_normalize}


def _call_here(fn: Callable[..., Any], *args: Any) -> Any:
    return fn(*args)


class StepUncertainty:
    """Measures one step right after it is generated: one shared set of Monte Carlo samples (each with its own
    seed) and token scores on the generating engine, then the metrics listed in `uncertainty.metrics` (MCSP,
    SE, SD, NLL, length). NLI and math-verify work
    runs through `call_main` (the runner's main thread: math-verify's timeouts need it, and UQLM's scorers
    are not thread-safe).
    The result is returned, never fed back to the agents."""

    def __init__(self, backend_for: Callable[[str], ModelBackend], call_main: Callable[..., Any] = _call_here) -> None:
        uq = get_settings().uncertainty
        self.backend_for, self.call_main, self.uq = backend_for, call_main, uq
        self.mcsp = MonteCarloSequenceProbability() if "mcsp" in uq.metrics else None
        self.se = SemanticEntropy() if "se" in uq.metrics else None
        self.sd = SemanticDensity() if "sd" in uq.metrics else None

    def measure(self, step: StepRecord, problem: str, marker: str | None = None) -> dict[str, Any]:
        engine = self.backend_for(step.agent_id)
        if self.uq.monte_carlo_enabled:
            MonteCarloSampler(engine).sample_steps([step])
        if self.uq.token_logprobs_needed:
            view = build_view(step, engine, marker)
        else:
            _require_tokens(step)
            view = UQView(step.text, [], [], [], len(step.token_ids), 0, 0)
        results = self.call_main(self._scores, view, problem)
        metrics = {name: value for name, (value, _) in results.items()}
        details = {name: d for name, (_, d) in results.items() if d}
        details["view"] = {
            "completion_marker_removed": view.rescored,
            "marker_only_texts": view.marker_only,
            "scored_text": view.text,
        }
        logger.info(
            "step uncertainty",
            extra={
                "problem_id": step.problem_id,
                "agent_id": step.agent_id,
                "round": step.round,
                "metrics": metrics,
            },
        )
        return {"metrics": metrics, "details": details}

    def _scores(self, view: UQView, problem: str) -> dict[str, tuple[float, dict[str, Any]]]:
        out: dict[str, tuple[float, dict[str, Any]]] = {}
        if self.mcsp:
            out["mcsp"] = self.mcsp.score(view, problem)
        if self.se:
            out["se"] = self.se.from_logprobs(view.text, view.mc_texts, view.token_logprobs, view.mc_logprobs)
        if self.sd:
            out["sd"] = self.sd.score(view, problem)
        if "nll" in self.uq.metrics:
            out["nll"] = (-sum(view.token_logprobs) / len(view.token_logprobs), {"normalized_by_length": True})
        if "length" in self.uq.metrics:
            out["length"] = (float(view.length), {})
        return {m: out[m] for m in self.uq.enabled_metrics()}


def step_metric_rows(trajectory: Trajectory) -> list[dict[str, Any]]:
    """The per-step metric rows: problem, agent, round, step, metrics."""
    return [
        {
            "problem_id": s.problem_id,
            "agent_id": s.agent_id,
            "round": s.round,
            "step_id": s.step_id,
            "metrics": s.metrics,
        }
        for a in trajectory.agents
        for s in a.steps
    ]
