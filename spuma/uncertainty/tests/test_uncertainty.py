import dataclasses
import math
import types

import numpy as np
import pytest
from uqlm.nli.nli import NLI

from spuma.agents.trajectory import AgentTrace, StepRecord, Trajectory
from spuma.models.backend import BackendCapabilities, BackendCapabilityError
from spuma.uncertainty.aggregation import aggregate_agent, mean, rms
from spuma.uncertainty.equivalence import AnswerAwareNLI, answer_equivalence
from spuma.uncertainty.estimators import (
    MC_SAMPLES_KEY,
    LabelOrderedNLI,
    MissingScoresError,
    MonteCarloSampler,
    MonteCarloSequenceProbability,
    SemanticDensity,
    SemanticEntropy,
    StepUncertainty,
    TokenLength,
    build_view,
    monte_carlo_seeds,
    step_metric_rows,
)
from spuma.uncertainty.features import extract_features

LPS = (-0.1, -0.2, -0.3, -0.4)


UNBOXED = "Let me check whether x = 2"


def _step(agent="solver", step_id=0, ids=(1, 2, 3, 4), lps=LPS, ents=(1.0, 2.0, 3.0, 4.0), text="t", **kw):
    return StepRecord(
        problem_id="p1",
        agent_id=agent,
        step_id=step_id,
        input_messages=[],
        prompt_ids=[9, 9],
        generation_params={"max_new_tokens": 16},
        seed=0,
        token_ids=list(ids),
        text=text,
        token_logprobs=list(lps) if lps is not None else None,
        token_entropies=list(ents) if ents is not None else None,
        **kw,
    )


@pytest.fixture
def with_uq(config, set_settings):
    """Install settings with `uncertainty.<section>` fields replaced, e.g. with_uq("nll", normalize_by_length=False)."""

    def install(section, **fields):
        uq = config.uncertainty
        part = dataclasses.replace(getattr(uq, section), **fields)
        return set_settings(dataclasses.replace(config, uncertainty=dataclasses.replace(uq, **{section: part})))

    return install


def _se(response, samples, lp=-0.5):
    """White-box SE with equal per-token log-probs: every text equally probable, so p_c = |c| / (n + 1)."""
    return SemanticEntropy().from_logprobs(response, samples, [lp] * 3, [[lp] * 3 for _ in samples])


def test_semantic_entropy_is_uqlm_white_box_on_response_plus_samples(rule_nli, with_uq):
    with_uq("semantic_entropy", equivalence="nli")
    samples = ["\\boxed{2}"] * 5 + ["so it is 2.0"] * 2 + ["\\boxed{3}"] * 2 + [UNBOXED]
    value, details = _se("t", samples)
    p = [1 / 11, 8 / 11, 2 / 11]
    assert value == pytest.approx(-sum(x * math.log(x) for x in p))
    assert details["semantic_negentropy"] == pytest.approx(1 - value / math.log(11))
    assert details["uqlm_scorer"] == "semantic_entropy (tokenprob)"
    assert ("t", "\\boxed{2}") in rule_nli.pairs


def test_semantic_entropy_is_zero_when_all_texts_agree():
    value, details = _se("so x = 4, \\boxed{4}", ["\\boxed{4}"] * 10)
    assert value == pytest.approx(0.0) and details["semantic_negentropy"] == pytest.approx(1.0)


def test_length_known_value_and_missing_tokens_raise():
    assert TokenLength().estimate(_step())[0] == 4
    with pytest.raises(MissingScoresError):
        TokenLength().estimate(_step(ids=(), lps=(), ents=()))


def _mc_sampled(backend, step=None):
    """A step with raw token log-probs and its Monte Carlo samples (log-probs at generation)."""
    step = step or _step(scored_by=backend.scorer_id)
    MonteCarloSampler(backend).sample_steps([step])
    return step


def _seq_prob(lps, normalize=True):
    return math.exp(sum(lps) / len(lps)) if normalize else math.exp(sum(lps))


def test_monte_carlo_samples_have_their_own_deterministic_seeds(scripted_backend_cls):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{2}"] * 10})
    step = _mc_sampled(backend)
    calls = backend.calls
    assert len(calls) == 10 and all(
        c.n == 1 and c.prompt_ids == step.prompt_ids and c.logprobs == "token" for c in calls
    )
    stored = step.uq_details[MC_SAMPLES_KEY]
    assert stored["seeds"] == [c.seed for c in calls] == monte_carlo_seeds(step) and len(set(stored["seeds"])) == 10
    assert stored["token_logprobs"][0] == [-0.5] * len("\\boxed{2}") and stored["scored_by"] == backend.scorer_id
    other_agent, other_round = _step(agent="enthusiast"), _step(round=2)
    assert not set(monte_carlo_seeds(other_agent)) & set(stored["seeds"])
    assert not set(monte_carlo_seeds(other_round)) & set(stored["seeds"])
    assert step.text == "t" and step.token_ids == [1, 2, 3, 4]


def test_monte_carlo_sampler_requires_token_logprobs():
    caps = BackendCapabilities(multiple_completions=True, batch_generation=True)
    with pytest.raises(BackendCapabilityError):
        MonteCarloSampler(type("B", (), {"name": "b", "capabilities": caps})())


@pytest.mark.parametrize("normalize", [True, False])
def test_mcsp_is_uqlm_white_box_mean_sequence_probability(scripted_backend_cls, with_uq, normalize):
    with_uq("monte_carlo", length_normalize=normalize)
    samples = ["\\boxed{2}"] * 6 + ["\\boxed{13}"] * 4
    backend = scripted_backend_cls([], samples={"": samples})
    view = build_view(_mc_sampled(backend), backend, None)
    value, details = MonteCarloSequenceProbability().score(view, "q")
    probs = [_seq_prob(LPS, normalize)] + [_seq_prob([-0.5] * len(t), normalize) for t in samples]
    assert value == pytest.approx(sum(probs) / len(probs)) and 0 < value <= 1
    assert details == {"uqlm_scorer": "monte_carlo_probability", "length_normalize": normalize}


def test_sd_is_uqlm_white_box_kernel_weighted_by_sample_probabilities(scripted_backend_cls, rule_nli):
    samples = ["\\boxed{2}"] * 6 + ["\\boxed{3}"] * 4
    backend = scripted_backend_cls([], samples={"": samples})
    step = _mc_sampled(backend, _step(text="x = 2", scored_by=backend.scorer_id))
    step.uq_details[MC_SAMPLES_KEY]["token_logprobs"] = [
        [-0.1] * len(t) if "2" in t else [-2.0] * len(t) for t in samples
    ]
    value, details = SemanticDensity().score(build_view(step, backend, None), "What is 1+1?")
    w_same, w_diff = 6 * math.exp(-0.1), 4 * math.exp(-2.0)
    assert value == pytest.approx(w_same / (w_same + w_diff)) and 0 <= value <= 1
    assert ("What is 1+1?\nx = 2", "What is 1+1?\n\\boxed{2}") in rule_nli.pairs
    assert details["uqlm_scorer"] == "semantic_density" and details["nli_prompt"] == "problem"


def test_view_strips_completion_marker_and_teacher_forces_the_rest(scripted_backend_cls):
    backend = scripted_backend_cls([], samples={"": ["Fine. TASK_DONE"] * 5 + ["Why?"] * 5}, logprob=-0.3)
    step = _mc_sampled(backend, _step(text="Looks right. TASK_DONE", scored_by=backend.scorer_id))
    view = build_view(step, backend, "TASK_DONE")
    assert view.text == "Looks right." and view.length == len("Looks right.") and view.rescored == 6
    assert view.token_logprobs == [-0.3] * len("Looks right.")
    assert backend.score_calls[0] == ([9, 9], [ord(c) for c in "Looks right."], False)
    assert view.mc_texts[:5] == ["Fine."] * 5 and view.mc_texts[5] == "Why?"
    assert step.text == "Looks right. TASK_DONE"
    untouched = build_view(_mc_sampled(backend, _step(text="Why?", scored_by=backend.scorer_id)), backend, "TASK_DONE")
    assert untouched.token_logprobs == list(LPS) and untouched.length == 4
    only = build_view(_mc_sampled(backend, _step(text="TASK_DONE", scored_by=backend.scorer_id)), backend, "TASK_DONE")
    assert only.text == "TASK_DONE" and only.token_logprobs == list(LPS) and only.marker_only == 1


def test_view_requires_response_logprobs(scripted_backend_cls):
    with pytest.raises(MissingScoresError, match="log-probabilities"):
        build_view(_step(lps=None), scripted_backend_cls([]), None)


def test_step_uncertainty_measures_four_metrics_with_scorers_on_the_main_thread(scripted_backend_cls, rule_nli):
    backend = scripted_backend_cls([], samples={"": ["\\boxed{2}"] * 10})
    served = []
    uq = StepUncertainty(lambda agent: backend, lambda fn, *a: served.append(fn.__name__) or fn(*a))
    step = _step(text="so \\boxed{2}", scored_by=backend.scorer_id)
    result = uq.measure(step, "What is 1+1?")
    assert list(result["metrics"]) == ["mcsp", "se", "nll", "length"] and served == ["_scores"]
    assert result["metrics"]["se"] == pytest.approx(0.0) and result["metrics"]["nll"] == pytest.approx(0.25)
    assert 0 < result["metrics"]["mcsp"] <= 1 and result["metrics"]["length"] == 4
    assert {"mcsp", "se", "view"} <= set(result["details"]) and "sd" not in result["details"] and uq.sd is None
    assert "se_samples" not in step.uq_details and len(step.uq_details[MC_SAMPLES_KEY]["texts"]) == 10
    assert len(backend.calls) == 10
    assert step.metrics == {}


def test_pipeline_se_is_white_box_tokenprob_entropy(scripted_backend_cls):
    """SE weights its clusters by the texts' sequence probabilities (UQLM tokenprob), not by counting texts."""
    samples = ["\\boxed{2}"] * 6 + ["\\boxed{3}"] * 4
    backend = scripted_backend_cls([], samples={"": samples})
    step = _mc_sampled(backend, _step(text="so \\boxed{2}", scored_by=backend.scorer_id))
    step.uq_details[MC_SAMPLES_KEY]["token_logprobs"] = [
        [-0.1] * len(t) if "2" in t else [-2.0] * len(t) for t in samples
    ]
    view = build_view(step, backend, None)
    value, details = SemanticEntropy().from_logprobs(view.text, view.mc_texts, view.token_logprobs, view.mc_logprobs)
    p2 = _seq_prob(LPS) + 6 * math.exp(-0.1)
    p3 = 4 * math.exp(-2.0)
    q = [p2 / (p2 + p3), p3 / (p2 + p3)]
    assert value == pytest.approx(-sum(x * math.log(x) for x in q))
    assert details["uqlm_scorer"] == "semantic_entropy (tokenprob)"


def test_enabling_sd_computes_it_and_disabled_metrics_are_skipped(with_metrics, scripted_backend_cls, rule_nli):
    with_metrics(["sd", "length"])
    backend = scripted_backend_cls([], samples={"": ["\\boxed{2}"] * 10})
    uq = StepUncertainty(lambda agent: backend)
    result = uq.measure(_step(text="so \\boxed{2}", scored_by=backend.scorer_id), "What is 1+1?")
    assert list(result["metrics"]) == ["sd", "length"] and result["metrics"]["sd"] == pytest.approx(1.0)
    assert uq.mcsp is None and uq.se is None


def test_length_only_draws_no_samples_and_needs_no_logprobs(with_metrics, scripted_backend_cls):
    with_metrics(["length"])
    backend = scripted_backend_cls([])
    step = _step(lps=None, ents=None)
    assert StepUncertainty(lambda agent: backend).measure(step, "q")["metrics"] == {"length": 4.0}
    assert backend.calls == [] and MC_SAMPLES_KEY not in step.uq_details


def test_one_semantic_entropy_and_density_scorer_per_measurement_engine(config, monkeypatch, scripted_backend_cls):
    loads = []
    monkeypatch.setattr(
        "spuma.uncertainty.estimators.load_uqlm_semantic_entropy", lambda: loads.append("se") or object()
    )
    monkeypatch.setattr("spuma.uncertainty.estimators.load_density_nli", lambda: loads.append("sd") or object())
    uq = StepUncertainty(lambda agent: scripted_backend_cls([]))
    assert loads == []
    assert uq.se.uq is uq.se.uq and loads == ["se"]


@pytest.mark.nli
def test_real_nli_semantic_entropy_clusters_paraphrases(with_uq, with_metrics):
    """Real UQLM SemanticEntropy + the configured NLI model: paraphrases of one answer score lower SE than
    two different answers."""
    with_uq("monte_carlo", num_generations=3)
    same, _ = _se(
        "Subtracting 3 from both sides gives 2x = 4, so x = 2.",
        ["The answer is x = 2.", "x equals 2.", "Solving gives x = 2."],
    )
    mixed, _ = _se("x = 2.", ["x = 2.", "x = 7.", "x = 7."])
    assert same < mixed and mixed > 0


def test_semantic_entropy_needs_n_samples_with_logprobs():
    with pytest.raises(MissingScoresError, match="exactly n=10"):
        _se("t", ["\\boxed{1}"] * 3)
    with pytest.raises(MissingScoresError, match="exactly n=10"):
        SemanticEntropy().from_logprobs("t", ["\\boxed{1}"] * 10, [-0.1], [[-0.1]] * 3)


@pytest.mark.nli
def test_real_nli_semantic_density_prefers_agreeing_samples(scripted_backend_cls, with_uq):
    """Real UQLM SemanticDensity + the configured NLI model: samples agreeing with the response score higher."""
    with_uq("monte_carlo", num_generations=3)
    response = "Subtracting 3 from both sides gives 2x = 4, so x = 2."
    values = []
    for samples in (
        ["The answer is x = 2.", "x equals 2.", "Solving gives x = 2."],
        ["The answer is x = 7.", "x equals 9.", "Solving gives x = -4."],
    ):
        backend = scripted_backend_cls([], samples={"": samples})
        step = _mc_sampled(backend, _step(text=response, scored_by=backend.scorer_id))
        values.append(SemanticDensity().score(build_view(step, backend, None), "Solve 2x + 3 = 7.")[0])
    assert values[0] > values[1]


def test_aggregation_known_values():
    assert rms([3, 4]) == pytest.approx(math.sqrt(12.5))
    assert mean([10, 20, 30]) == 20
    with pytest.raises(ValueError):
        rms([])


def _trajectory():
    def trace(agent, metrics_list):
        steps = [_step(agent=agent, step_id=i, round=i + 1, metrics=m) for i, m in enumerate(metrics_list)]
        return AgentTrace(agent_id=agent, role=agent, model="m", steps=steps)

    m = lambda mc, se, nll, ln: {"mcsp": mc, "se": se, "nll": nll, "length": ln}  # noqa: E731
    return Trajectory(
        problem_id="p1",
        problem="q",
        final_answer_text="\\boxed{2}",
        agents=[
            trace("mathematician", [m(0.3, 0, 0.4, 10), m(0.4, 0, 0.6, 30)]),
            trace("enthusiast", [m(0.9, 1, 0.5, 5)]),
        ],
    )


def test_features_are_four_metrics_per_agent(config):
    feats = extract_features(_trajectory())
    assert list(feats) == config.features.order and len(feats) == 8
    assert feats["mathematician_mcsp"] == pytest.approx(math.sqrt((0.09 + 0.16) / 2))
    assert feats["mathematician_length"] == 20
    assert feats["enthusiast_se"] == 1 and feats["enthusiast_nll"] == pytest.approx(0.5)
    assert feats["mathematician_nll"] == pytest.approx(math.sqrt((0.16 + 0.36) / 2))


def test_step_metric_rows_carry_agent_and_round():
    rows = step_metric_rows(_trajectory())
    assert [(r["agent_id"], r["round"]) for r in rows] == [
        ("mathematician", 1),
        ("mathematician", 2),
        ("enthusiast", 1),
    ]


def test_aggregate_unknown_strategy(config, set_settings):
    with pytest.raises(ValueError):
        set_settings(
            dataclasses.replace(
                config, aggregation=dataclasses.replace(config.aggregation, **{"semantic_density": "median"})
            )
        )
        aggregate_agent(_trajectory().agents[0], ["sd"])


def _fake_nli(monkeypatch, id2label, probs):
    """NLI.__init__ / predict_batch replaced: no model download, a model with labels `id2label`."""

    def init(self, **kwargs):
        self.label_mapping, self.max_tokens = ["contradiction", "neutral", "entailment"], 512
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(id2label=id2label))

    monkeypatch.setattr(NLI, "__init__", init)
    monkeypatch.setattr(NLI, "predict_batch", lambda self, pairs: np.array([probs]))


def test_label_ordered_nli_maps_any_label_order_to_uqlms(monkeypatch):
    _fake_nli(monkeypatch, {0: "entailment", 1: "neutral", 2: "contradiction"}, [0.9, 0.08, 0.02])
    nli = LabelOrderedNLI(max_tokens=2048, nli_model_name="any")
    assert nli.max_tokens == 2048
    assert nli.predict_batch([("a", "b")]).tolist() == [[0.02, 0.08, 0.9]]
    _fake_nli(monkeypatch, {0: "yes", 1: "no", 2: "maybe"}, [0.3, 0.3, 0.4])
    with pytest.raises(ValueError, match="lack"):
        LabelOrderedNLI(nli_model_name="any")


def test_answer_aware_equivalence_uses_math_verify_then_nli(rule_nli):
    """Default `equivalence: answer_aware`: boxed answers by math-verify, unanswered texts by the NLI model."""
    step = "Long derivation ... so the answer is \\boxed{\\frac{1}{2}}"
    samples = (
        ["Different words entirely, \\boxed{0.5}"] * 3
        + ["\\boxed{1/2}"]
        + ["Hence \\boxed{2}"] * 2
        + [UNBOXED] * 2
        + ["ran out of tokens at step 7", "ran out of tokens at step 9"]
    )
    value, details = _se(step, samples)
    p = [5 / 11, 2 / 11, 2 / 11, 1 / 11, 1 / 11]
    assert value == pytest.approx(-sum(x * math.log(x) for x in p))
    assert details["equivalence"] == "answer_aware"
    assert rule_nli.pairs and all("boxed" not in a + b for a, b in rule_nli.pairs)


def test_answer_aware_loads_nli_only_when_needed():
    loads = []
    nli = AnswerAwareNLI(lambda: loads.append(1))
    assert [
        r["entailment"] for r in nli.get_nli_results_batch([("\\boxed{3}", "so \\boxed{3.0}"), ("\\boxed{3}", "x")])
    ] == [True, False]
    assert loads == [] and nli.counts == {"math_verify": 1, "answer_vs_none": 1, "nli": 0}


@pytest.mark.parametrize(
    ("dataset", "a", "b", "same"),
    [
        ("math", "\\frac{1}{2}", "0.5", True),
        ("math", "32 \\text{ miles per hour}", "32", True),
        ("math", "0 \\text{ and } 4", "0, 4", True),
        ("math", "\\begin{pmatrix} 0 \\\\ -5 \\end{pmatrix}", "\\begin{pmatrix} 0 \\\\ 5 \\end{pmatrix}", False),
        ("morehopqa", "\\frac{1}{2}", "0.5", True),
        ("morehopqa", "\\text{Sunday}", "Sunday", True),
        ("morehopqa", "the Beatles", "Beatles", True),
        ("morehopqa", "May 5, 1990", "5 May 1990", True),
        ("morehopqa", "May 1990", "May 5, 1990", False),
        ("morehopqa", "listen", "silent", False),
        ("morehopqa", "John Smith", "Smith John", False),
    ],
)
def test_boxed_answers_are_compared_per_dataset(dataset, a, b, same):
    assert answer_equivalence(dataset)(a, b) is same
