import numpy as np
import pytest

from spuma.evaluation.risk import RISK_MODELS, risk_model_spec
from spuma.spn.model import ClassConditionalSPN


def test_registry_builds_trains_and_reloads_spn(config, set_settings, tmp_path):
    spec = risk_model_spec("spn")
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.4).astype(int)
    Z = rng.normal(size=(200, 8)) + y[:, None]
    set_settings(config.replace(spn=config.spn.model_copy(update={"min_instances_slice": 500})))
    model = spec.build().fit(Z, y)
    assert isinstance(model, ClassConditionalSPN) and set(model.summary()) >= {"log_prior", "features"}
    model.save(tmp_path / f"spn{spec.suffix}")
    assert np.allclose(spec.load(tmp_path / f"spn{spec.suffix}").predict_error_proba(Z), model.predict_error_proba(Z))


def test_unknown_risk_model():
    assert list(RISK_MODELS) == ["spn"]
    with pytest.raises(ValueError, match="registered"):
        risk_model_spec("xgboost")


def test_config_only_accepts_registered_models(raw_config):
    from pydantic import ValidationError

    from spuma.config import Settings

    raw_config["evaluation"]["risk_models"] = ["xgboost"]
    with pytest.raises(ValidationError):
        Settings.from_dict(raw_config)
