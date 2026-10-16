import numpy as np
import pytest

from spuma.evaluation.metrics import (
    METRICS,
    auarc,
    aurc,
    bootstrap_ci,
    evaluate,
    expected_calibration_error,
    risk_coverage,
)


def test_risk_coverage_rejects_highest_risk_first():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.2, 0.8])
    cov, risk = risk_coverage(y, p)
    assert cov.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert risk.tolist() == pytest.approx([0, 0, 1 / 3, 0.5])
    assert aurc(y, p) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert auarc(y, p) == pytest.approx(1 - aurc(y, p))


def test_auarc_ordering_and_ties():
    y = np.array([0, 1, 0, 1])
    good, bad = np.array([0.1, 0.9, 0.2, 0.8]), np.array([0.9, 0.1, 0.8, 0.2])
    assert auarc(y, good) > auarc(y, bad)
    assert aurc(y, np.zeros(4)) == pytest.approx(0.5)


def test_auroc_and_single_class():
    assert METRICS["auroc"](np.array([0, 1]), np.array([0.2, 0.8])) == 1.0
    assert np.isnan(METRICS["auroc"](np.array([1, 1]), np.array([0.2, 0.8])))


def test_brier_nll_accuracy_ece():
    y, p = np.array([0, 1]), np.array([0.25, 0.75])
    assert METRICS["brier"](y, p) == pytest.approx(0.0625)
    assert METRICS["nll"](y, p) == pytest.approx(-np.log(0.75))
    assert METRICS["accuracy"](y, p) == 0.5
    assert expected_calibration_error(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([0, 1]), np.array([0.5, 0.5]), bins=2) == 0.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.random(200) * 0.4, 0, 1)
    lo, hi = bootstrap_ci(y, p, METRICS["auroc"], np.arange(200), 200, 0.95, seed=0)
    assert lo <= METRICS["auroc"](y, p) <= hi


def test_evaluate_and_unknown_metric():
    y, p = np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.3, 0.6])
    res = evaluate(
        y,
        p,
        ["auroc", "auarc"],
        groups=np.array(list("abcd")),
        bootstrap={"iterations": 20, "confidence": 0.9, "seed": 0},
    )
    assert res["auroc"]["value"] == 1.0 and "ci_low" in res["auarc"]
    with pytest.raises(ValueError):
        evaluate(y, p, ["nope"])
