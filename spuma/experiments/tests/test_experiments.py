import dataclasses

import pytest
import yaml

from spuma.config import get_settings
from spuma.experiments import __main__ as cli
from spuma.experiments import base as base_module
from spuma.experiments import registry
from spuma.experiments.base import Experiment


def test_math_experiment_is_registered_with_its_settings():
    math = registry.EXPERIMENTS["math"]
    settings = math.activate()
    assert settings.experiment.name == "ma_qwen_math" and settings.dataset.name == "math"
    assert [a.id for a in settings.agents.enabled] == ["solver", "reviewer"]


@dataclasses.dataclass(frozen=True)
class _Recorded(Experiment):
    """Runs nothing: records which settings were active when its runner ran."""

    log: list = dataclasses.field(default_factory=list, compare=False)
    fail: bool = False

    def runner(self):
        exp = self

        class Runner:
            run_id = "test"

            def run(self, phases, limit):
                exp.log.append((get_settings().experiment.name, phases, limit))
                if exp.fail:
                    raise RuntimeError("boom")

        return Runner()


@pytest.fixture
def experiments(tmp_path, raw_config, monkeypatch):
    """Two experiments `a`, `b` with their own settings files (they differ in experiment.name)."""
    log, released = [], []
    made = {}
    for name in ("a", "b"):
        raw = {**raw_config, "experiment": {**raw_config["experiment"], "name": f"exp_{name}"}}
        raw["logging"] = {**raw_config["logging"], "file": False, "console": False}
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(raw))
        made[name] = _Recorded(name=name, config_file=path, log=log)
    monkeypatch.setattr(registry, "EXPERIMENTS", made)
    monkeypatch.setattr(cli, "EXPERIMENTS", made)
    monkeypatch.setattr(base_module, "release_backends", lambda: released.append(1))
    return made, log, released


def test_all_experiments_run_by_default_each_with_its_own_settings(experiments):
    _, log, released = experiments
    assert cli.main(["--phase", "evaluate", "--limit", "3"]) == 0
    assert log == [("exp_a", ["evaluate"], 3), ("exp_b", ["evaluate"], 3)]
    assert len(released) == 2  # engines freed after each experiment


def test_selected_experiment_only(experiments):
    _, log, _ = experiments
    assert cli.main(["--experiment", "b"]) == 0
    assert log == [("exp_b", None, None)]
    with pytest.raises(SystemExit):
        cli.main(["--experiment", "unknown"])


def test_a_failing_experiment_does_not_stop_the_others(experiments, monkeypatch):
    made, log, released = experiments
    monkeypatch.setitem(made, "a", dataclasses.replace(made["a"], fail=True))
    assert cli.main([]) == 1
    assert [name for name, _, _ in log] == ["exp_a", "exp_b"] and len(released) == 2


def test_missing_settings_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        Experiment(name="x", config_file=tmp_path / "missing.yaml").activate()
