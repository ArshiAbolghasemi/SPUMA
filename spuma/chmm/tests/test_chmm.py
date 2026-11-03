import dataclasses
import json
import os

import numpy as np
import pytest
import yaml

from spuma.chmm import experiment as chmm_experiment
from spuma.chmm.config import CHMMConfig, DistanceConfig, HMMConfig
from spuma.chmm.distances import LexicalDistance, preceding, situational_distances, token_f1
from spuma.chmm.experiment import STANDALONE, CHMMAggregation, chmm_method
from spuma.chmm.hmm import SituationalHMM, state_coefficients, step_weights
from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, get_settings
from spuma.dataset.base import Problem
from spuma.experiments.runner import ExperimentRunner
from spuma.experiments.tests.test_e2e import Engines, FakeDataset, fake_capabilities  # noqa: F401


def test_alpha_zero_gives_uniform_weights_and_weights_sum_to_one():
    gamma = np.random.default_rng(0).dirichlet(np.ones(3), size=7)
    assert np.allclose(step_weights(gamma, 0.0), 1 / 7)
    for a in (0.5, 1, 2, 4):
        assert state_coefficients(a).tolist() == [1, 1 + a, 1 + 2 * a]
        w = step_weights(gamma, a)
        assert w.sum() == pytest.approx(1) and (w > 0).all()


def test_alpha_must_be_positive_in_config():
    with pytest.raises(ValueError, match="alpha = 0"):
        CHMMConfig(alphas=[0.0, 1.0])


def test_hmm_orders_states_by_deviation_and_fills_missing_with_training_mean():
    rng = np.random.default_rng(1)
    means = [0.1, 0.5, 0.9]
    seqs = []
    for _ in range(150):
        s = rng.integers(0, 3, 6)
        x = np.array([[means[k], means[k]] for k in s]) + rng.normal(0, 0.03, (6, 2))
        x[0, 1] = np.nan
        seqs.append(x)
    hmm = SituationalHMM(HMMConfig()).fit(seqs)
    described = hmm.describe()
    assert np.allclose(np.array(described["means"])[:, 0], means, atol=0.05)
    assert np.all(np.diff(hmm.deviation) > 0)
    # a missing gap is the training-mean gap: standardized to 0, an average gap
    assert hmm.fill[1] == pytest.approx(np.nanmean(np.concatenate(seqs)[:, 1]))
    assert np.allclose(hmm._scaled([np.array([[0.5, np.nan]])])[0][0, 1], 0)
    g = hmm.posteriors([np.array([[0.5, np.nan], [0.9, 0.9], [0.1, 0.1]]), np.array([[0.5, np.nan], [0.5, 0.5]])])
    assert [len(x) for x in g] == [3, 2] and all(np.allclose(x.sum(1), 1) for x in g)
    assert g[0][1].argmax() == 2 and g[0][2].argmax() == 0 and g[1][1].argmax() == 1
    # imputation, not marginalization: a missing gap reads exactly as the training-mean gap
    missing, filled = hmm.posteriors([np.array([[0.9, np.nan]]), np.array([[0.9, hmm.fill[1]]])])
    assert np.allclose(missing, filled)


def test_hmm_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    seqs = [rng.random((4, 2)) for _ in range(40)]
    hmm = SituationalHMM(HMMConfig(n_iter=20)).fit(seqs)
    hmm.save(tmp_path / "chmm.pkl")
    again = SituationalHMM.load(tmp_path / "chmm.pkl")
    assert np.allclose(np.concatenate(again.posteriors(seqs)), np.concatenate(hmm.posteriors(seqs)))


def test_preceding_response_feedback_and_star():
    assert preceding(["m", "e", "m", "e"]) == [None, 0, 1, 2]
    # star: coordinator c, workers a/b; a worker's relevant response is the coordinator's, and vice versa
    assert preceding(["c", "a", "b", "c", "a"]) == [None, 0, 1, 2, 3]
    assert preceding(["a", "a"]) == [None, None]


def test_situational_distances_first_gap_is_missing():
    d = situational_distances(
        LexicalDistance(DistanceConfig()), "what is two plus two", ["m", "e"], ["two plus two is 4", "is it 4"]
    )
    assert d[0]["interaction_gap"] is None and 0 <= d[1]["interaction_gap"] <= 1
    assert d[0]["question_drift"] < d[1]["question_drift"]
    assert token_f1("The 840.", "840") == 1.0 and token_f1("12", "13") == 0.0


@pytest.fixture
def experiment(raw_config, tmp_path):
    """A small fake MATH experiment run end to end (generate .. evaluate), CHMM included (joint SPNs only)."""
    raw_config["class_conditional_spn"]["enabled"] = False
    return _run_experiment(raw_config, tmp_path)


def _run_experiment(raw, tmp_path):
    raw["experiment"].update({"name": "id_math", "output_dir": str(tmp_path), "concurrent_problems": 1})
    raw["agents"]["max_rounds"] = 3
    raw["evaluation"]["bootstrap"]["iterations"] = 20
    raw["spn"]["min_instances_slice"] = 20
    raw["model"]["backend"], raw["model"]["scoring_backend"] = "transformers", "same"
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw))
    os.environ[SETTINGS_FILE_ENV] = str(path)
    get_settings.cache_clear()
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    runner.run()
    return runner


def test_chmm_runs_in_the_experiment_and_evaluates_frozen(experiment):
    runner = experiment
    assert get_settings().chmm.enabled and get_settings().chmm.distance.estimator == "lexical"
    aggregation = CHMMAggregation(runner)
    out = runner.store.path(aggregation.out)
    ev = runner.store.read_json(f"{aggregation.out}/evaluation.json")
    spns = ["rms_spn", *(chmm_method(a) for a in get_settings().chmm.alphas)]
    methods = [*STANDALONE, *(x for m in spns for x in (m, f"{m}_entropy"))]
    assert list(ev["methods"]) == methods and ev["frozen_from"] is None
    assert np.isnan(ev["methods"]["rms_spn_entropy"]["brier"]["value"])  # entropy is a ranking score only
    assert set(runner.store.read_json("summary.json")["chmm"]) == set(methods)
    assert ev["checks"]["max_weight_sum_deviation"] < 1e-9
    assert ev["lambdas"] == {"0.5": [1.0, 1.5, 2.0], "1": [1.0, 2.0, 3.0], "2": [1.0, 3.0, 5.0], "4": [1.0, 5.0, 9.0]}
    for name in ("chmm.json", "steps.jsonl", "features.json", "diagnostics.json", "results.md", "alpha.png"):
        assert (out / name).is_file()
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_text().splitlines()]
    assert all(s["interaction_gap"] is None for s in steps if s["t"] == 0)
    for pid in {s["problem_id"] for s in steps}:
        assert sum(s["weights"]["1"] for s in steps if s["problem_id"] == pid) == pytest.approx(1)
    rows = runner.store.read_json(f"{aggregation.out}/features.json")
    per_agent = [f"{g}_{m}" for g in ("solver", "reviewer") for m in ("se", "mcsp", "nll", "length")]
    assert all(list(v) == per_agent for r in rows for v in r["features"].values())

    # evaluate never fits: the frozen CHMM, SPNs and calibrators give the same results
    before = {p.name: p.read_bytes() for p in (out / "models").iterdir()}
    again = aggregation.evaluate(frozen_from=out)
    assert {p.name: p.read_bytes() for p in (out / "models").iterdir()} == before
    assert json.dumps(again["methods"]) == json.dumps(ev["methods"])  # NaN-aware (entropy has no Brier/ECE)


def test_class_conditional_spn_is_tuned_on_validation_and_compared_with_the_joint_spn(raw_config, tmp_path):
    raw_config["class_conditional_spn"] = {
        "enabled": True,
        "leaves": ["normal", "histogram"],
        "rdc_threshold": [0.3],
        "min_instances_slice": [20],
        "variance_floor": [0.01, 0.1],
        "histogram_smoothing": [1.0],
    }
    runner = _run_experiment(raw_config, tmp_path)
    aggregation = CHMMAggregation(runner)
    ev = runner.store.read_json(f"{aggregation.out}/evaluation.json")
    for m in aggregation.spn_methods:
        status = runner.store.read_json(f"{aggregation.out}/models/{m}_cc_status.json")
        assert status["model"] == "class_conditional" and len(status["tuning"]) == 3
        assert status["validation_rows"] > 0 and set(status["structure"]) == {"correct", "wrong"}
        assert {"auroc", "auprc", "auarc", "brier", "ece"} <= set(ev["methods"][f"{m}_cc"])
        assert set(ev["class_conditional_comparisons"][f"{m}_cc-{m}"]) == set(ev["methods"][m])
    assert "class-conditional SPN minus joint SPN" in (runner.store.path(aggregation.out) / "results.md").read_text()


class EvalOnlyDataset:
    """No training data; test questions the fake model answers, two of them unlabeled (blind)."""

    def load(self):
        test = [
            Problem(
                f"eval/test/{k}",
                f"What is {k % 9 + 2}+{k // 3 + 5}?",
                None if k < 2 else f"\\boxed{{{k % 9 + 2 + k // 3 + 5}}}",
                "test",
            )
            for k in range(30)
        ]
        return {"train": [], "validation": [], "test": test}


def test_eval_only_dataset_skips_training_and_evaluates_the_frozen_source(experiment, tmp_path, monkeypatch):
    source_runner = experiment
    source_file = tmp_path / "source.yaml"
    configs = {"src": source_file}
    monkeypatch.setattr(chmm_experiment, "experiment_config", configs.__getitem__)
    source_dir = source_runner.store.path(CHMMAggregation(source_runner).out)
    before = {p.name: p.read_bytes() for p in (source_dir / "models").iterdir()}

    raw = yaml.safe_load(source_file.read_text())
    raw["experiment"]["name"] = "target"
    raw["dataset"].update({"train_samples": 0, "validation_samples": 0, "test_samples": 30})
    raw["chmm"]["frozen_from"] = ["src"]
    target_file = tmp_path / "target.yaml"
    target_file.write_text(yaml.safe_dump(raw))
    os.environ[SETTINGS_FILE_ENV] = str(target_file)
    get_settings.cache_clear()
    engines = Engines()
    runner = ExperimentRunner(EvalOnlyDataset(), engines.factory, release=engines.release)
    runner.run()

    d = runner.scored_dir()
    labels = [r["label"] for r in runner.store.read_json(f"{d}/features.json")]
    assert labels.count(None) == 2
    aggregation = CHMMAggregation(runner)
    assert runner.store.read_json(f"{aggregation.out}/status.json")["status"] == "skipped"
    summary = runner.store.read_json("summary.json")
    assert "chmm" not in summary and list(summary["chmm_frozen"]) == ["src"]
    frozen = runner.store.read_json(f"{aggregation.out}/frozen-src/evaluation.json")
    assert frozen["n_test"] == 28 and frozen["frozen_from"] == str(source_dir)
    spns = ["rms_spn", *(chmm_method(a) for a in get_settings().chmm.alphas)]
    assert set(frozen["methods"]) == {*STANDALONE, *spns, *(f"{m}_entropy" for m in spns)}
    assert {p.name: p.read_bytes() for p in (source_dir / "models").iterdir()} == before

    configs["target"] = target_file
    rows = chmm_experiment.consolidate(["src", "target"])
    assert {(r["experiment"], r["protocol"]) for r in rows} == {("src", "dataset_specific"), ("target", "frozen:src")}
    assert len(rows) == 28  # (4 standalone + 5 SPNs + their 5 entropies) x 2 protocols


def test_feature_groups_aggregate_per_agent_per_response_or_pooled(monkeypatch):
    def problem(agents, nodes, values, weights):
        steps = [
            {"agent_index": a, "node": n, "metrics": dict(zip(("se", "mcsp", "nll", "length"), [v] * 4, strict=True))}
            for a, n, v in zip(agents, nodes, values, strict=True)
        ]
        return {"steps": steps, "weights": {1.0: np.array(weights)}}

    def aggregation(layout):
        agg = CHMMAggregation.__new__(CHMMAggregation)
        agg.config = dataclasses.replace(get_settings().chmm, feature_layout=layout)
        return agg

    # feedback loop, 2 rounds: solver says 1 then 3, reviewer 2 then 4
    fb = problem([0, 1, 0, 1], [None] * 4, [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4])
    agg = aggregation("per_agent")
    assert agg.feature_names[:2] == ["solver_se", "solver_mcsp"] and len(agg.feature_names) == 8
    rms = agg._aggregate([fb], "rms_spn")[0]
    assert np.allclose(rms[[0, 4]], [np.sqrt((1 + 9) / 2), np.sqrt((4 + 16) / 2)])
    chmm = agg._aggregate([fb], "chmm_a1_spn")[0]
    assert np.allclose(chmm[[0, 4]], [(0.1 * 1 + 0.3 * 3) * 4 / 2, (0.2 * 2 + 0.4 * 4) * 4 / 2])
    uniform = problem([0, 1, 0, 1], [None] * 4, [1.0, 2.0, 3.0, 4.0], [0.25] * 4)
    assert np.allclose(agg._aggregate([uniform], "chmm_a1_spn")[0][[0, 4]], [2.0, 3.0])  # per-agent means
    # pooled: the old single weighted mean over all responses
    pooled = aggregation("pooled")
    assert pooled.feature_names == ["se", "mcsp", "nll", "length"]
    assert np.allclose(pooled._aggregate([fb], "chmm_a1_spn")[0][0], 0.1 + 0.4 + 0.9 + 1.6)

    # star: every node on its own, scaled by its relative weight T * w_t
    monkeypatch.setenv(SETTINGS_FILE_ENV, str(PROJECT_ROOT / "config" / "star" / "math_ma_qwen2_5_7b_star.yaml"))
    get_settings.cache_clear()
    nodes = ["star_delegate", "analyst", "star_request_verification", "verifier", "star_finalize"]
    star = problem([0, 1, 0, 2, 0], nodes, [1.0, 2.0, 3.0, 4.0, 5.0], [0.1, 0.1, 0.2, 0.2, 0.4])
    agg = aggregation("per_agent")
    assert len(agg.feature_names) == 20 and agg.feature_names[0] == "star_delegate_se"
    assert np.allclose(agg._aggregate([star], "rms_spn")[0][::4], [1, 2, 3, 4, 5])
    assert np.allclose(agg._aggregate([star], "chmm_a1_spn")[0][::4], [0.5, 1.0, 3.0, 4.0, 10.0])
