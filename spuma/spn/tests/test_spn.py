import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from spuma.evaluation.metrics import binary_entropy
from spuma.spn.model import ClassConditionalSPN


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    Z = rng.normal(size=(n, 8)) + 1.5 * y[:, None]
    return Z, y


@pytest.fixture
def make_spn(config, set_settings):
    """A ClassConditionalSPN built from the settings, with these `spn:` overrides installed."""

    def make(**spn):
        set_settings(dataclasses.replace(config, spn=dataclasses.replace(config.spn, **spn)))
        return ClassConditionalSPN()

    return make


@pytest.fixture
def fitted(make_spn):
    Z, y = _synthetic()
    return make_spn(min_instances_slice=1000).fit(Z, y), Z, y


def test_posterior_is_probability_and_separates(fitted):
    model, Z, y = fitted
    p = model.predict_error_proba(Z)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > 0.8 and p[y == 0].mean() < 0.2


def test_posterior_normalizes(fitted):
    model, Z, _ = fitted
    lj = model.log_joint(Z[:10])
    post = np.exp(lj - np.logaddexp(lj[:, :1], lj[:, 1:]))
    assert np.allclose(post.sum(1), 1.0)


def test_matches_analytic_naive_bayes(fitted):
    # min_instances_slice above n forces fully factorized Gaussian leaves: an exact Gaussian naive Bayes
    # (SPFlow MLE uses the unbiased variance)
    model, Z, y = fitted
    Zs = (Z - model.mean) / model.std
    ll = []
    for c in (0, 1):
        d = Zs[y == c]
        ll.append(norm.logpdf(Zs[:5], d.mean(0), d.std(0, ddof=1)).sum(1))
    lj = np.stack(ll, 1) + model.log_prior
    expected = np.exp(lj[:, 1] - np.logaddexp(lj[:, 0], lj[:, 1]))
    assert np.allclose(model.predict_error_proba(Z[:5]), expected, atol=1e-6)


def test_marginalization_equals_dropping_feature(fitted):
    model, Z, _ = fitted
    Zn = Z[:5].copy()
    Zn[:, 3] = np.nan
    a = model.predict_error_proba(Zn)
    b = model.predict_error_proba(Z[:5], marginalize=[model.feature_names[3]])
    assert np.allclose(a, b) and np.isfinite(a).all()
    all_missing = model.predict_error_proba(np.full((1, 8), np.nan))
    assert np.allclose(all_missing, np.exp(model.log_prior[1]))  # prior when nothing is observed


def test_binary_entropy():
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(np.log(2))
    assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-9


def test_learned_structure_and_serialization(tmp_path, make_spn, config):
    Z, y = _synthetic(600, seed=1)
    Z[:, 1] = Z[:, 0] + 0.1 * np.random.default_rng(2).normal(size=len(Z))  # dependent pair
    model = make_spn(min_instances_slice=50).fit(Z, y)
    p = model.predict_error_proba(Z[:20])
    model.save(tmp_path / "spn.pt")
    loaded = ClassConditionalSPN.load(tmp_path / "spn.pt")
    assert np.allclose(loaded.predict_error_proba(Z[:20]), p)
    assert loaded.feature_names == config.features.order and len(loaded.feature_names) == 8


def test_requires_both_classes(make_spn):
    Z, _ = _synthetic(20)
    with pytest.raises(ValueError):
        make_spn().fit(Z, np.zeros(20, dtype=int))


def test_constant_features_do_not_break_structure_learning(make_spn):
    Z, y = _synthetic(300, seed=3)
    Z[:, 5] = 0.0  # e.g. semantic entropy 0 everywhere
    Z[y == 1, 6] = 1.0  # constant within one class only
    model = make_spn(min_instances_slice=30).fit(Z, y)
    assert np.isfinite(model.predict_error_proba(Z)).all()
