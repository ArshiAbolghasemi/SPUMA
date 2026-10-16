import dataclasses
import math
import types

import numpy as np
import pytest
from langchain_core.messages import HumanMessage
from uqlm.nli.nli import NLI

from spuma.agents.trajectory import AgentTrace, StepRecord, Trajectory
from spuma.models.backend import BackendCapabilities, BackendCapabilityError, GenerationRequest, scorer_id
from spuma.models.config import GenerationConfig, ModelConfig
from spuma.uncertainty.aggregation import aggregate_agent, mean, rms
from spuma.uncertainty.equivalence import AnswerAwareNLI
from spuma.uncertainty.estimators import (
    LabelOrderedNLI,
    MissingScoresError,
    NegativeLogLikelihood,
    PredictiveEntropy,
    SemanticEntropy,
    SemanticSampler,
    TokenLength,
    TokenScorer,
    build_estimators,
    build_token_scorers,
    score_trajectory,
    step_metric_rows,
)
from spuma.uncertainty.features import extract_features

UNBOXED = "Let me check whether x = 2"  # no boxed answer, ends in 2


def _step(agent="solver", step_id=0, ids=(1, 2, 3, 4), lps=(-0.1, -0.2, -0.3, -0.4), ents=(1.0, 2.0, 3.0, 4.0), **kw):
    return StepRecord(
        problem_id="p1",
        agent_id=agent,
        step_id=step_id,
        input_messages=[],
        prompt_ids=[9, 9],
        generation_params={"max_new_tokens": 16},
        seed=0,
        token_ids=list(ids),
        text="t",
        token_logprobs=list(lps) if lps is not None else None,
        token_entropies=list(ents) if ents is not None else None,
        **kw,
    )


def test_pe_nll_length_known_values(with_uq):
    s = _step()
    assert PredictiveEntropy().estimate(s)[0] == pytest.approx(2.5)
    assert NegativeLogLikelihood().estimate(s)[0] == pytest.approx(0.25)
    with_uq("nll", normalize_by_length=False)
    assert NegativeLogLikelihood().estimate(s)[0] == pytest.approx(1.0)
    assert TokenLength().estimate(s)[0] == 4


def test_missing_scores_raise_instead_of_fabricating():
    with pytest.raises(MissingScoresError):
        PredictiveEntropy().estimate(_step(ents=None))
    with pytest.raises(MissingScoresError):
        NegativeLogLikelihood().estimate(_step(lps=None))
    with pytest.raises(MissingScoresError):
        TokenLength().estimate(_step(ids=(), lps=(), ents=()))


def test_real_backend_pe_nll_alignment(tiny_backend):

    prompt = tiny_backend.render([HumanMessage("Solve for x")])
    request = GenerationRequest(prompt, GenerationConfig(temperature=0.7, max_new_tokens=6), seed=3, logprobs="full")
    (g,) = tiny_backend.generate(request)
    s = _step(ids=g.token_ids, lps=g.token_logprobs, ents=g.token_entropies)
    scores = tiny_backend.score(prompt, g.token_ids, full_vocabulary=True)
    lps, ents = scores.token_logprobs, scores.token_entropies
    assert PredictiveEntropy().estimate(s)[0] == pytest.approx(sum(ents) / len(ents))
    assert NegativeLogLikelihood().estimate(s)[0] == pytest.approx(-sum(lps) / len(lps))


@pytest.fixture
def with_uq(config, set_settings):
    """Install settings with `uncertainty.<section>` fields replaced, e.g. with_uq("nll", normalize_by_length=False)."""

    def install(section, **fields):
        uq = config.uncertainty
        part = dataclasses.replace(getattr(uq, section), **fields)
        return set_settings(dataclasses.replace(config, uncertainty=dataclasses.replace(uq, **{section: part})))

    return install


def _sampled(backend, step=None):
    step = step or _step()
    SemanticSampler(backend).sample_steps([step])
    return step


def test_semantic_entropy_uses_uqlm_on_step_text_plus_samples(scripted_backend_cls, rule_nli, with_uq):
    # equivalence: nli (rule NLI stand-in); step text "t" has no number, so it forms its own cluster
    with_uq("semantic_entropy", equivalence="nli")
    samples = ["\\boxed{2}"] * 5 + ["so it is 2.0"] * 2 + ["\\boxed{3}"] * 2 + [UNBOXED]
    backend = scripted_backend_cls([], samples={"": samples})
    step = _sampled(backend)
    value, details = SemanticEntropy().estimate(step)
    # 11 texts: {t}, {2 x5, 2.0 x2, unboxed text ending in "x = 2"}, {3 x2}  ->  1/11, 8/11, 2/11
    p = [1 / 11, 8 / 11, 2 / 11]
    assert value == pytest.approx(-sum(x * math.log(x) for x in p))
    assert details["semantic_negentropy"] == pytest.approx(1 - value / math.log(11))
    assert ("t", "\\boxed{2}") in rule_nli.pairs  # the step's own text is clustered with its samples
    (call,) = backend.calls
    assert call.n == 10 and call.prompt_ids == step.prompt_ids and call.params.top_k is None
    assert call.logprobs == "none"  # samples are never scored
    stored = step.uq_details["se_samples"]
    assert stored["policy"]["nli_model"]
    assert step.token_ids == [1, 2, 3, 4] and step.text == "t"  # main trajectory untouched


def test_semantic_samples_are_cached_and_batched(scripted_backend_cls, with_uq):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{4}"] * 10})
    sampler = SemanticSampler(backend)
    steps = [_step(step_id=0), _step(step_id=1)]
    sampler.sample_steps(steps)
    sampler.sample_steps(steps)  # already sampled: no new backend calls
    assert len(backend.calls) == 2 and not any(sampler.needs_samples(s) for s in steps)
    # step text "t" vs ten identical "\\boxed{4}" samples: clusters 1/11 and 10/11
    expected = -(1 / 11 * math.log(1 / 11) + 10 / 11 * math.log(10 / 11))
    assert SemanticEntropy().estimate(steps[0])[0] == pytest.approx(expected)
    # a different SE configuration invalidates the stored samples
    with_uq("semantic_entropy", temperature=1.0)
    assert SemanticSampler(backend).needs_samples(steps[0])


def test_semantic_entropy_from_fixed_texts_is_backend_independent():
    """Fixed texts give the same SE whichever backend sampled them; all agree -> 0."""
    value, details = SemanticEntropy().from_texts("so x = 4, \\boxed{4}", ["\\boxed{4}"] * 10)
    assert value == pytest.approx(0.0) and details["semantic_negentropy"] == pytest.approx(1.0)


def test_trajectory_scored_in_one_uqlm_call(config, scripted_backend_cls, monkeypatch):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{1}"] * 10})
    traj = _trajectory()
    sampler = SemanticSampler(backend)
    for a in traj.agents:
        for s in a.steps:
            s.metrics, s.scored_by = {}, backend.scorer_id
        sampler.sample_steps(a.steps)
    ests = build_estimators()
    se = next(e for e in ests["solver"] if e.name == "se")
    calls = []
    real = se.uq.score
    monkeypatch.setattr(se.uq, "score", lambda **kw: calls.append(len(kw["responses"])) or real(**kw))
    score_trajectory(traj, ests, build_token_scorers(lambda m, k: pytest.fail("scored")))
    assert calls == [3]  # 2 solver steps + 1 reviewer step, one call


def test_one_nli_model_shared_by_agents(config, monkeypatch):
    loads = []
    monkeypatch.setattr("spuma.uncertainty.estimators.load_uqlm_semantic_entropy", lambda: loads.append(1) or object())
    ests = build_estimators()
    se = [e for e in ests["solver"] if e.name == "se"][0]
    assert se is [e for e in ests["reviewer"] if e.name == "se"][0] and loads == []  # lazy
    assert se.uq is se.uq and len(loads) == 1


@pytest.mark.nli
def test_real_deberta_nli_clusters_paraphrases(with_uq):
    """Real BlackBoxUQ + DeBERTa-MNLI: paraphrases of one answer score lower SE than two different answers."""
    with_uq("semantic_entropy", n=3)
    se = SemanticEntropy()
    same, _ = se.from_texts(
        "Subtracting 3 from both sides gives 2x = 4, so x = 2.",
        ["The answer is x = 2.", "x equals 2.", "Solving gives x = 2."],
    )
    mixed, _ = se.from_texts("x = 2.", ["x = 2.", "x = 7.", "x = 7."])
    assert same < mixed and mixed > 0


def test_semantic_entropy_requires_stored_samples():
    with pytest.raises(MissingScoresError, match="generate phase"):
        SemanticEntropy().estimate(_step())


def test_sampler_requires_multiple_completions():
    backend = type("B", (), {"name": "nosample", "capabilities": BackendCapabilities()})()
    with pytest.raises(BackendCapabilityError):
        SemanticSampler(backend)


def test_semantic_entropy_rejects_short_sample_list(scripted_backend_cls):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{1}"] * 3})
    with pytest.raises(MissingScoresError):
        _sampled(backend)


def test_token_scorer_skips_steps_scored_by_same_engine(scripted_backend_cls):
    backend = scripted_backend_cls([])
    step = _step(scored_by=backend.scorer_id)
    scorer = TokenScorer(backend.scorer_id, lambda: pytest.fail("engine must not load"), full_vocabulary=True)
    scorer.ensure(step)
    assert step.token_logprobs == [-0.1, -0.2, -0.3, -0.4]


def test_token_scorer_rescores_with_other_engine(scripted_backend_cls):
    """Hybrid mode: steps generated (unscored) by vLLM are teacher-forced on Transformers."""
    tf = scripted_backend_cls([], logprob=-0.7)
    step = _step(lps=None, ents=None, scored_by=None, generation_meta={"compat": tf.info.compatibility_key()})
    TokenScorer(tf.scorer_id, lambda: tf, full_vocabulary=True).ensure(step)
    assert tf.score_calls == [([9, 9], [1, 2, 3, 4], True)]  # exact saved prompt + continuation ids
    assert step.scored_by == tf.scorer_id and step.token_logprobs == [-0.7] * 4 and len(step.token_entropies) == 4
    assert NegativeLogLikelihood().estimate(step)[0] == pytest.approx(0.7)


def test_token_scorer_rejects_incompatible_engine(scripted_backend_cls):
    tf = scripted_backend_cls([])
    step = _step(lps=None, ents=None, generation_meta={"compat": {"model": "other"}})
    with pytest.raises(BackendCapabilityError, match="differs"):
        TokenScorer(tf.scorer_id, lambda: tf, full_vocabulary=False).ensure(step)


def test_build_token_scorers_share_engine(config, set_settings):
    scorers = build_token_scorers(lambda m, kind: pytest.fail("lazy"))
    assert scorers["solver"] is scorers["reviewer"]
    assert scorers["solver"].scorer == scorer_id(config.model, "transformers") and scorers["solver"].full_vocabulary
    set_settings(dataclasses.replace(config, model=ModelConfig(backend="vllm", scoring_backend="transformers")))
    assert build_token_scorers(lambda m, k: None)["solver"].scorer.startswith("transformers:")


def test_aggregation_known_values():
    assert rms([3, 4]) == pytest.approx(math.sqrt(12.5))
    assert mean([10, 20, 30]) == 20
    with pytest.raises(ValueError):
        rms([])


def _trajectory():
    def trace(agent, metrics_list):
        steps = [_step(agent=agent, step_id=i, metrics=m) for i, m in enumerate(metrics_list)]
        return AgentTrace(agent_id=agent, role=agent, model="m", steps=steps)

    m = lambda pe, se, nll, ln: {"pe": pe, "se": se, "nll": nll, "length": ln}  # noqa: E731
    return Trajectory(
        problem_id="p1",
        problem="q",
        final_answer_text="\\boxed{2}",
        agents=[
            trace("solver", [m(3, 0, 1, 10), m(4, 0, 1, 30)]),
            trace("reviewer", [m(1, 1, 2, 5)]),
        ],
    )


def test_features_order_and_values(config):
    feats = extract_features(_trajectory())
    assert list(feats) == config.features.order
    assert feats["solver_pe"] == pytest.approx(math.sqrt(12.5))
    assert feats["solver_length"] == 20  # mean, not RMS
    assert feats["reviewer_nll"] == 2


def test_aggregate_unknown_strategy(config, set_settings):
    with pytest.raises(ValueError):
        set_settings(
            dataclasses.replace(config, aggregation=dataclasses.replace(config.aggregation, **{"nll": "median"}))
        )
        aggregate_agent(_trajectory().agents[0], ["nll"])


def test_score_trajectory_and_rows(config, scripted_backend_cls):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{1}"] * 10})
    traj = _trajectory()
    sampler = SemanticSampler(backend)
    for a in traj.agents:
        for s in a.steps:
            s.metrics = {}
            s.scored_by = backend.scorer_id
        sampler.sample_steps(a.steps)
    scorers = build_token_scorers(lambda m, kind: pytest.fail("already scored at generation"))
    scored = score_trajectory(traj, build_estimators(), scorers)
    rows = step_metric_rows(scored)
    assert set(rows[0]["metrics"]) == {"pe", "se", "nll", "length"}
    assert traj.agents[0].steps[0].metrics == {}  # input not mutated
    assert len(backend.calls) == 3  # one 10-sample SE request per step


def _fake_nli(monkeypatch, id2label, probs):
    """NLI.__init__ / predict_batch replaced: no model download, a model with labels `id2label`."""

    def init(self, **kwargs):
        self.label_mapping, self.max_tokens = ["contradiction", "neutral", "entailment"], 512
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(id2label=id2label))

    monkeypatch.setattr(NLI, "__init__", init)
    monkeypatch.setattr(NLI, "predict_batch", lambda self, pairs: np.array([probs]))


def test_label_ordered_nli_maps_any_label_order_to_uqlms(monkeypatch):
    # ModernBERT-style order: the model says "entailment" with 0.9
    _fake_nli(monkeypatch, {0: "entailment", 1: "neutral", 2: "contradiction"}, [0.9, 0.08, 0.02])
    nli = LabelOrderedNLI(max_tokens=2048, nli_model_name="any")
    assert nli.max_tokens == 2048
    assert nli.predict_batch([("a", "b")]).tolist() == [[0.02, 0.08, 0.9]]  # UQLM reads column 2 as entailment
    _fake_nli(monkeypatch, {0: "yes", 1: "no", 2: "maybe"}, [0.3, 0.3, 0.4])
    with pytest.raises(ValueError, match="lack"):
        LabelOrderedNLI(nli_model_name="any")


def test_answer_aware_equivalence_uses_math_verify_then_nli(rule_nli):
    """Default `equivalence: answer_aware`: boxed answers by math-verify, unanswered texts by the NLI model."""
    step = "Long derivation ... so the answer is \\boxed{\\frac{1}{2}}"
    samples = (
        ["Different words entirely, \\boxed{0.5}"] * 3  # same answer, other text: math-verify merges
        + ["\\boxed{1/2}"]
        + ["Hence \\boxed{2}"] * 2  # a different answer
        + [UNBOXED] * 2  # no boxed answer: NLI decides (rule NLI: both end in 2 -> same)
        + ["ran out of tokens at step 7", "ran out of tokens at step 9"]  # NLI: different final numbers
    )
    value, details = SemanticEntropy().from_texts(step, samples)
    p = [5 / 11, 2 / 11, 2 / 11, 1 / 11, 1 / 11]
    assert value == pytest.approx(-sum(x * math.log(x) for x in p))
    assert details["equivalence"] == "answer_aware"
    assert rule_nli.pairs and all("boxed" not in a + b for a, b in rule_nli.pairs)  # NLI never sees answer pairs


def test_answer_aware_loads_nli_only_when_needed():
    loads = []
    nli = AnswerAwareNLI(lambda: loads.append(1))
    assert [
        r["entailment"] for r in nli.get_nli_results_batch([("\\boxed{3}", "so \\boxed{3.0}"), ("\\boxed{3}", "x")])
    ] == [True, False]
    assert loads == [] and nli.counts == {"math_verify": 1, "answer_vs_none": 1, "nli": 0}
