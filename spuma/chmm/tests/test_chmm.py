import json
import os

import numpy as np
import pytest
import yaml

from spuma.chmm.config import CHMMConfig, DistanceConfig, HMMConfig
from spuma.chmm.distances import LexicalDistance, preceding, situational_distances, token_f1
from spuma.chmm.experiment import STANDALONE, CHMMAggregation, chmm_method
from spuma.chmm.hmm import SituationalHMM, state_coefficients, step_weights
from spuma.config import SETTINGS_FILE_ENV, get_settings
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
    """A small fake MATH experiment run end to end (generate .. evaluate), CHMM included."""
    raw = raw_config
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
    methods = [*STANDALONE, "rms_spn", *(chmm_method(a) for a in get_settings().chmm.alphas)]
    assert list(ev["methods"]) == methods and ev["frozen_from"] is None
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
    assert all(len(v) == 4 for r in rows for v in r["features"].values())

    # evaluate never fits: the frozen CHMM, SPNs and calibrators give the same results
    before = {p.name: p.read_bytes() for p in (out / "models").iterdir()}
    again = aggregation.evaluate(frozen_from=out)
    assert {p.name: p.read_bytes() for p in (out / "models").iterdir()} == before
    assert again["methods"] == ev["methods"]
