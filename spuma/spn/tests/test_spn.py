import dataclasses
import inspect

import numpy as np
import pytest
import torch
from spflow.meta.data.scope import Scope
from spflow.modules.ops.cat import Cat
from spflow.modules.products import Product
from spflow.modules.sums import Sum

from spuma.evaluation.metrics import auroc, binary_entropy
from spuma.spn import model as spn_model
from spuma.spn.config import HistogramConfig, SPNConfig
from spuma.spn.leaves import (
    BinaryIndicatorLeaf,
    FlooredNormal,
    HistogramLeaf,
    LeafSettings,
    histogram_edges,
    leaf_settings,
)
from spuma.spn.model import ClassConditionalSPN, JointSPN, partition_keeping_label, partition_with_constants

NAN = float("nan")


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    Z = rng.normal(size=(n, 8)) + 1.5 * y[:, None]
    return Z, y


@pytest.fixture
def make_spn(config, set_settings):
    """A joint SPN built from the settings, with these `spn:` overrides installed."""

    def make(**spn):
        set_settings(dataclasses.replace(config, spn=dataclasses.replace(SPNConfig(), **spn)))  # scalar defaults
        return JointSPN()

    return make


@pytest.fixture(params=["histogram", "normal", "laplace"])
def fitted(request, make_spn):
    Z, y = _synthetic()
    return make_spn(leaf=request.param).fit(Z, y), Z, y


def _settings(**edges):
    return leaf_settings(
        LeafSettings(label_smoothing=1.0, bin_smoothing=1.0, edges={int(k[1:]): v for k, v in edges.items()})
    )


def test_indicators_encode_one_binary_variable():
    pos, neg = BinaryIndicatorLeaf.indicators(torch.tensor([1.0, 0.0, NAN]))
    assert pos.tolist() == [1, 0, 1] and neg.tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        BinaryIndicatorLeaf.indicators(torch.tensor([2.0]))


def test_indicator_leaf_normalizes_and_marginalizes():
    with _settings():
        leaf = BinaryIndicatorLeaf(Scope([0]))
    leaf.maximum_likelihood_estimation(torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64))
    p = {v: leaf.log_likelihood(torch.tensor([[v]], dtype=torch.float64)).exp().item() for v in (1.0, 0.0)}
    assert p[1.0] == pytest.approx((2 + 1) / (3 + 2))
    assert p[1.0] + p[0.0] == pytest.approx(1.0)
    assert leaf.log_likelihood(torch.tensor([[NAN]], dtype=torch.float64)).item() == pytest.approx(0.0, abs=1e-12)


def _integral(leaf, edges):
    """Exact over the bins (constant density), numerical over the exponential tails."""
    f = lambda v: leaf.log_likelihood(torch.as_tensor(v, dtype=torch.float64)[:, None]).exp().detach().numpy().ravel()  # noqa: E731
    mids = (edges[:-1] + edges[1:]) / 2
    inner = float((f(mids) * np.diff(edges)).sum())
    lo = np.linspace(edges[0] - 80, edges[0], 400001)[:-1]
    hi = np.linspace(edges[-1], edges[-1] + 80, 400001)[1:]
    return inner + np.trapezoid(f(lo), lo) + np.trapezoid(f(hi), hi)


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_histogram_is_a_density_with_no_zero_cells(strategy):
    x = np.r_[np.zeros(80), np.random.default_rng(0).exponential(1.0, 120)]
    edges = histogram_edges(x, 8, strategy)
    assert len(edges) == 9 and np.all(np.diff(edges) > 0)
    with _settings(v0=torch.tensor(edges)):
        leaf = HistogramLeaf(Scope([0]))
    leaf.maximum_likelihood_estimation(torch.tensor(x)[:, None])
    assert _integral(leaf, edges) == pytest.approx(1.0, abs=1e-4)
    far = leaf.log_likelihood(torch.tensor([[edges[0] - 5.0], [edges[-1] + 5.0]], dtype=torch.float64))
    assert torch.isfinite(far).all()
    assert leaf.log_likelihood(torch.tensor([[NAN]], dtype=torch.float64)).item() == 0.0


def test_histogram_mle_is_smoothed_counts():
    edges = np.array([0.0, 1.0, 2.0])
    with _settings(v0=torch.tensor(edges)):
        leaf = HistogramLeaf(Scope([0]))
    leaf.maximum_likelihood_estimation(torch.tensor([[0.5], [0.5], [1.5]], dtype=torch.float64))
    expected = np.log([1 / 7, 3 / 7, 2 / 7, 1 / 7])
    assert np.allclose(leaf.log_probs.detach().numpy().ravel(), expected)
    assert leaf.log_likelihood(torch.tensor([[0.2]], dtype=torch.float64)).item() == pytest.approx(np.log(3 / 7))


def test_histogram_config_validates():
    with pytest.raises(ValueError):
        HistogramConfig(strategy="kde")
    with pytest.raises(ValueError):
        HistogramConfig(smoothing=0.0)


def _children(module):
    inputs = module.inputs
    return list(inputs.inputs) if isinstance(inputs, Cat) else [inputs]


def test_joint_structure_is_valid_and_covers_the_label(make_spn):
    Z, y = _synthetic(600, seed=1)
    Z[:, 1] = Z[:, 0] + 0.1 * np.random.default_rng(2).normal(size=len(Z))
    model = make_spn(min_instances_slice=50).fit(Z, y)
    assert set(model.root.scope.query) == set(range(9))
    sums = products = 0
    for m in model.root.modules():
        if isinstance(m, Sum):
            sums += 1
            assert len({tuple(sorted(c.scope.query)) for c in _children(m)}) == 1
        elif isinstance(m, Product):
            products += 1
            scopes = [set(c.scope.query) for c in _children(m)]
            assert sum(map(len, scopes)) == len(set().union(*scopes))
    assert sums >= 1 and products >= 1
    labels = [m for m in model.root.modules() if isinstance(m, BinaryIndicatorLeaf)]
    assert labels and all(list(m.scope.query) == [8] for m in labels)


def test_posterior_is_probability_and_separates(fitted):
    model, Z, y = fitted
    p = model.predict_error_proba(Z)
    assert ((p >= 0) & (p <= 1)).all()
    assert p[y == 1].mean() > 0.6 and p[y == 0].mean() < 0.4
    assert np.allclose(model.posterior_entropy(Z), binary_entropy(p))


def test_label_marginalization_is_sum_over_labels(fitted):
    model, Z, _ = fitted
    lj = model.log_joint(Z[:20])
    assert np.allclose(model.log_marginal(Z[:20]), np.logaddexp(lj[:, 0], lj[:, 1]))


def test_feature_marginalization_and_prior(fitted):
    model, Z, _ = fitted
    Zn = Z[:5].copy()
    Zn[:, 3] = np.nan
    a = model.predict_error_proba(Zn)
    b = model.predict_error_proba(Z[:5], marginalize=[model.feature_names[3]])
    assert np.allclose(a, b) and np.isfinite(a).all()
    nothing = np.full((1, 8), np.nan)
    assert model.log_marginal(nothing)[0] == pytest.approx(0.0, abs=1e-9)
    assert np.exp(model.log_prior()).sum() == pytest.approx(1.0)
    assert model.predict_error_proba(nothing)[0] == pytest.approx(np.exp(model.log_prior()[1]))


def test_joint_density_integrates_to_one(make_spn):
    Z, y = _synthetic(400, seed=4)
    model = make_spn(leaf="normal", min_instances_slice=100).fit(Z, y)
    grid = np.linspace(-60, 60, 240001)
    Zg = np.full((len(grid), 8), np.nan)
    Zg[:, 0] = grid * model.std[0] + model.mean[0]
    density = np.exp(model.log_marginal(Zg)) / model.std[0]
    assert np.trapezoid(density, Zg[:, 0]) == pytest.approx(1.0, abs=1e-4)


def test_no_label_leakage_at_inference(make_spn):
    Z, y = _synthetic(400, seed=5)
    Z[1], y[0], y[1] = Z[0], 0, 1
    model = make_spn().fit(Z, y)
    for fn in (model.predict_error_proba, model.posterior_entropy, model.log_joint):
        assert "y" not in inspect.signature(fn).parameters
    p = model.predict_error_proba(Z[:2])
    assert p[0] == p[1]
    shuffled = np.random.default_rng(0).permutation(len(Z))
    assert np.allclose(model.predict_error_proba(Z[shuffled]), model.predict_error_proba(Z)[shuffled])


def test_posterior_normalizes(make_spn):
    Z, y = _synthetic()
    model = make_spn().fit(Z, y)
    lj = model.log_joint(Z[:10])
    post = np.exp(lj - np.logaddexp(lj[:, :1], lj[:, 1:]))
    assert np.allclose(post.sum(1), 1.0)


@pytest.mark.parametrize("leaf", ["histogram", "normal"])
def test_serialization_round_trip(tmp_path, make_spn, config, leaf):
    Z, y = _synthetic(600, seed=1)
    model = make_spn(leaf=leaf).fit(Z, y)
    p = model.predict_error_proba(Z[:20])
    model.save(tmp_path / "spn.pt")
    loaded = JointSPN.load(tmp_path / "spn.pt")
    assert np.allclose(loaded.predict_error_proba(Z[:20]), p)
    assert loaded.feature_names == config.features.order and loaded.config.leaf == leaf
    torch.save({"kind": "class_conditional"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="class_conditional SPN, not a joint SPN"):
        JointSPN.load(tmp_path / "old.pt")


def test_requires_both_classes(make_spn):
    Z, _ = _synthetic(20)
    with pytest.raises(ValueError):
        make_spn().fit(Z, np.zeros(20, dtype=int))


@pytest.mark.parametrize("leaf", ["histogram", "normal"])
def test_constant_features_do_not_break_structure_learning(make_spn, leaf):
    Z, y = _synthetic(300, seed=3)
    Z[:, 5] = 0.0
    Z[y == 1, 6] = 1.0
    model = make_spn(leaf=leaf, min_instances_slice=30).fit(Z, y)
    assert np.isfinite(model.predict_error_proba(Z)).all()


def test_label_is_never_split_off_alone_while_it_varies():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(float)
    x = np.c_[rng.normal(size=300) + 2 * y, rng.normal(size=300), rng.normal(size=300)]
    data = torch.tensor(np.c_[x, y])
    plain = partition_with_constants(data, threshold=0.99)
    assert int((plain == plain[3]).sum()) == 1
    kept = partition_keeping_label(data, threshold=0.99)
    assert kept[3] == kept[0] and int((kept == kept[3]).sum()) == 2
    constant = data.clone()
    constant[:, 3] = 0.0
    ids = partition_keeping_label(constant, threshold=0.99)
    assert int((ids == ids[3]).sum()) == 1


def test_label_placement_survives_an_rdc_failure(monkeypatch):
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(float)
    x = np.c_[rng.normal(size=300), rng.normal(size=300) + 2 * y, rng.normal(size=300)]
    data = torch.tensor(np.c_[x, y])

    def failing_rdc(a, b):
        raise RuntimeError("max(): Expected reduction dim to be specified for input.numel() == 0")

    monkeypatch.setattr(spn_model, "rdc", failing_rdc)
    kept = partition_keeping_label(data, threshold=0.99)
    assert kept[3] == kept[1]  # joined with the most correlated feature, not split off


def test_floored_normal_keeps_variance_above_floor():
    with leaf_settings(LeafSettings(label_smoothing=1.0, bin_smoothing=1.0, edges={}, variance_floor=0.25)):
        leaf = FlooredNormal(Scope([0]))
    leaf.maximum_likelihood_estimation(torch.full((10, 1), 3.0, dtype=torch.float64))
    assert leaf.scale.item() == pytest.approx(0.5)


GRID = {
    "leaf": ["normal", "histogram"],
    "rdc_threshold": [0.2, 0.3],
    "min_instances_slice": [50],
    "variance_floor": [0.01, 0.1],
}


def test_grid_candidates_cross_each_leaf_with_its_regularizer():
    spn = SPNConfig(**GRID, histogram=HistogramConfig(smoothing=[1.0, 5.0]))
    got = {(c.leaf, c.rdc_threshold, c.variance_floor, c.histogram.smoothing) for c in spn.candidates()}
    assert len(spn.candidates()) == len(got) == 2 * 2 + 2 * 2
    assert ("histogram", 0.2, 0.0, 5.0) in got and ("normal", 0.3, 0.1, 1.0) in got
    assert SPNConfig().candidates() == [SPNConfig()]  # scalars: no tuning
    with pytest.raises(ValueError):
        SPNConfig(rdc_threshold=[0.3, 1.5])


@pytest.mark.parametrize("cls", [JointSPN, ClassConditionalSPN])
def test_grid_is_tuned_on_validation_brier_and_checkpointed(cls, config, set_settings, tmp_path):
    set_settings(dataclasses.replace(config, spn=SPNConfig(**GRID)))
    Z, y = _synthetic()
    Zv, yv = _synthetic(200, seed=1)
    model = cls().fit(Z, y, Zv, yv)
    assert len(model.tuning) == 6 and model.config in SPNConfig(**GRID).candidates()
    best = min(t["validation_brier"] for t in model.tuning)
    assert np.mean((model.predict_error_proba(Zv) - yv) ** 2) == pytest.approx(best)
    assert auroc(yv, model.predict_error_proba(Zv)) > 0.9
    model.save(tmp_path / "m.pt")
    loaded = cls.load(tmp_path / "m.pt")
    assert np.allclose(loaded.predict_error_proba(Zv), model.predict_error_proba(Zv))
    assert loaded.hyperparameters() == model.hyperparameters() and loaded.config == model.config
    # no validation rows: the first grid point, untuned
    assert cls().fit(Z, y).config == SPNConfig(**GRID).candidates()[0]


def test_a_failing_grid_candidate_is_skipped(config, set_settings, monkeypatch):
    set_settings(dataclasses.replace(config, spn=SPNConfig(**GRID)))
    Z, y = _synthetic()
    Zv, yv = _synthetic(200, seed=1)
    bad = SPNConfig(**GRID).candidates()[0]
    real_fit = JointSPN._fit

    def fit(self, Z, y):
        if self.config == bad:
            raise RuntimeError("RDC failed")
        return real_fit(self, Z, y)

    monkeypatch.setattr(JointSPN, "_fit", fit)
    model = JointSPN().fit(Z, y, Zv, yv)
    failed = [t for t in model.tuning if "error" in t]
    assert len(failed) == 1 and failed[0]["validation_brier"] == float("inf") and model.config != bad
    monkeypatch.setattr(JointSPN, "_fit", lambda self, Z, y: (_ for _ in ()).throw(RuntimeError("RDC failed")))
    with pytest.raises(RuntimeError, match="every joint SPN candidate failed"):
        JointSPN().fit(Z, y, Zv, yv)


def test_class_conditional_spn_bayes_rule_and_empirical_prior(make_spn, tmp_path):
    make_spn(leaf="normal")
    Z, y = _synthetic()
    Zv, _ = _synthetic(200, seed=1)
    model = ClassConditionalSPN().fit(Z, y)
    assert np.allclose(np.exp(model.log_prior()), [(y == 0).mean(), (y == 1).mean()])  # raw training shares
    data = model._features(Zv, None)
    ll = [model.root["spns"][c].log_likelihood(data).reshape(len(data), -1)[:, 0].detach().numpy() for c in (0, 1)]
    a, b = np.log((y == 1).mean()) + ll[1], np.log((y == 0).mean()) + ll[0]
    assert np.allclose(model.predict_error_proba(Zv), 1 / (1 + np.exp(b - a)))
    model.save(tmp_path / "cc.pt")
    with pytest.raises(ValueError, match="class_conditional"):
        JointSPN.load(tmp_path / "cc.pt")
