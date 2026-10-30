import dataclasses
import os

import pytest
import yaml

from spuma.config import SETTINGS_FILE_ENV, get_settings
from spuma.dataset.base import Problem
from spuma.dataset.config import DatasetConfig
from spuma.experiments.runner import ExperimentRunner
from spuma.experiments.tests.test_e2e import Engines, FakeDataset, _is_wrong, fake_capabilities  # noqa: F401
from spuma.ood import __main__ as cli
from spuma.ood.config import OODAgentPrompt, OODConfig, load_ood_config
from spuma.ood.evaluation import OODEvaluation, sha256
from spuma.ood.registry import OOD_EVALUATIONS

METRICS = ["auroc", "auprc", "auarc", "aurc", "brier", "ece", "nll", "accuracy"]


class FakeOODDataset:
    """Test-only problems the fake model can answer; numeric references as MoreHopQA stores them."""

    def load(self):
        test = []
        for k in range(40):
            a, b = k % 13 + 20, k // 2 + 31
            test.append(
                Problem(f"morehopqa/test/{k}", f"What is {a}+{b}?", str(a + b), "test", {"answer_type": "number"})
            )
        return {"train": [], "validation": [], "test": test}


@pytest.fixture
def source(raw_config, tmp_path):
    """A source experiment settings file (small and fast) and its trained in-distribution SPN."""
    raw = raw_config
    raw["experiment"].update({"name": "id_math", "output_dir": str(tmp_path), "concurrent_problems": 1})
    raw["agents"]["max_rounds"] = 3
    raw["evaluation"]["bootstrap"]["iterations"] = 50
    raw["evaluation"]["risk_models"] = ["spn"]  # the source checkpoint the OOD evaluation loads
    raw["spn"]["min_instances_slice"] = 20
    raw["model"]["backend"], raw["model"]["scoring_backend"] = "transformers", "same"
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw))
    os.environ[SETTINGS_FILE_ENV] = str(path)
    get_settings.cache_clear()
    engines = Engines()
    runner = ExperimentRunner(FakeDataset(), engines.factory, release=engines.release)
    runner.run()
    checkpoint = runner.store.path(runner.model_path("spn"))
    return path, checkpoint


def _ood(source, tmp_path, **fields):
    path, checkpoint = source
    dataset = DatasetConfig(
        name="morehopqa", data_files="x.parquet", subsets=[], train_samples=0, validation_samples=0, test_samples=40
    )
    return OODConfig(
        name="ood_test", source_config=path, checkpoint=checkpoint, dataset=dataset, output_dir=tmp_path, **fields
    )


def test_frozen_spn_is_evaluated_on_ood_problems(source, tmp_path, fake_capabilities):  # noqa: F811
    config = _ood(source, tmp_path)
    checkpoint = config.checkpoint
    digest, mtime = sha256(checkpoint), checkpoint.stat().st_mtime_ns
    engines = Engines()
    evaluation = OODEvaluation(
        config, {"dataset": FakeOODDataset(), "backend_factory": engines.factory, "release": engines.release}
    )
    result = evaluation.run()

    assert sha256(checkpoint) == digest == result["checkpoint"]["sha256"] and checkpoint.stat().st_mtime_ns == mtime
    root = tmp_path / "ood_test"
    assert (root / "settings.yaml").is_file() and yaml.safe_load((root / "settings.yaml").read_text())["dataset"][
        "name"
    ] == "morehopqa"
    assert not list(root.rglob("models")), "OOD evaluation must not train any model"

    labels = [_is_wrong(k % 13 + 20, k // 2 + 31) for k in range(40)]
    assert result["n"] == 40 and result["accuracy"] == pytest.approx(1 - sum(labels) / 40)
    assert set(result["models"]["spn"]) == set(METRICS) and "ci_low" in result["models"]["spn"]["auarc"]
    assert set(result["models"]["spn_posterior_entropy"]) == {"auroc", "auprc", "auarc", "aurc"}
    assert len(result["predictions"]) == 40 and all(0 <= r["p_incorrect"] <= 1 for r in result["predictions"])
    assert {r["label"] for r in result["predictions"]} == {0, 1}

    comparison = result["id_comparison"]
    assert set(comparison["metrics"]) == set(METRICS) and comparison["id_n"] == 30
    auroc = comparison["metrics"]["auroc"]
    assert auroc["delta"] == pytest.approx(auroc["ood"] - auroc["id"])
    assert set(comparison["features"]) == set(get_settings().features.order)

    store = root / next(p.name for p in root.iterdir() if p.name.startswith("generation-"))
    scored = next(store.glob("scored-*"))
    assert (scored / "ood_evaluation.json").is_file() and (scored / "steps.jsonl").is_file()
    assert (scored / "step_metrics.json").is_file() and (root / "ood_summary.json").is_file()

    again = OODEvaluation(config, {"dataset": FakeOODDataset(), "backend_factory": Engines().factory}).run(
        phases=["evaluate"]
    )
    assert again["models"]["spn"]["auroc"]["value"] == result["models"]["spn"]["auroc"]["value"]


def test_source_settings_must_reproduce_the_checkpoint_pipeline(source, tmp_path):
    path, _ = source
    raw = yaml.safe_load(path.read_text())
    raw["model"]["generation"]["temperature"] = 0.9
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(raw))
    config = dataclasses.replace(_ood(source, tmp_path), source_config=changed)
    with pytest.raises(ValueError, match="reproduce the checkpoint"):
        OODEvaluation(config).activate()


def test_ood_config_file_is_validated(tmp_path):
    good = load_ood_config(OOD_EVALUATIONS["ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop"])
    assert good.name == "ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop"
    assert good.source_config.name == "math_ma_qwen2_5_7b_feedbackloop.yaml"
    assert good.dataset.name == "morehopqa" and good.dataset.test_samples == 400 and good.checkpoint.is_absolute()
    assert {k: v.role for k, v in good.agents.items()} == {"mathematician": "solver", "enthusiast": "reviewer"}
    bad = tmp_path / "x.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "ood": {
                    "name": "x",
                    "source_config": "a",
                    "checkpoint": "b",
                    "dataset": {"name": "morehopqa", "train_samples": 5},
                }
            }
        )
    )
    with pytest.raises(ValueError, match="test-only"):
        load_ood_config(bad)
    bad.write_text(
        yaml.safe_dump(
            {"ood": {"name": "x", "source_config": "a", "checkpoint": "b", "dataset": {}, "metrics": ["f1"]}}
        )
    )
    with pytest.raises(ValueError, match="ood.metrics"):
        load_ood_config(bad)
    bad.write_text(
        yaml.safe_dump(
            {
                "ood": {
                    "name": "other",
                    "source_config": "a",
                    "checkpoint": "b",
                    "dataset": {"train_samples": 0, "validation_samples": 0},
                }
            }
        )
    )
    with pytest.raises(ValueError, match="must equal the file name"):
        load_ood_config(bad)


def test_cli_lists_and_runs_by_name(monkeypatch, capsys):
    assert cli.main(["--list"]) == 0
    assert "ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop" in capsys.readouterr().out
    ran = []
    monkeypatch.setattr(
        cli.OODEvaluation, "run", lambda self, phases, limit: ran.append((self.config.name, phases, limit))
    )
    assert (
        cli.main(["--ood", "ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop", "--phase", "evaluate", "--limit", "3"]) == 0
    )
    assert ran == [("ood_morehopqa_math_ma_qwen2_5_7b_feedbackloop", ["evaluate"], 3)]
    with pytest.raises(SystemExit):
        cli.main(["--ood", "unknown"])


def test_ood_agents_replace_roles_and_prompts_after_the_fingerprint_check(source, tmp_path):
    prompts = {
        "mathematician": OODAgentPrompt(role="solver", system_prompt="Answer with \\boxed{}."),
        "enthusiast": OODAgentPrompt(role="reviewer", system_prompt="Review.", user_template="{problem}\n{previous}"),
    }
    OODEvaluation(_ood(source, tmp_path, agents=prompts)).activate()
    agents = {a.id: a for a in get_settings().agents.enabled}
    assert (agents["mathematician"].role, agents["mathematician"].system_prompt) == ("solver", "Answer with \\boxed{}.")
    assert agents["enthusiast"].user_template == "{problem}\n{previous}"
    assert get_settings().agents.workflow == "feedback"
    with pytest.raises(ValueError, match="must name the source agents"):
        OODEvaluation(_ood(source, tmp_path, agents={"solver": prompts["mathematician"]})).activate()
