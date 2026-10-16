import subprocess
import sys

import pytest

from spuma.config import PROJECT_ROOT
from spuma.experiments.runner import ExperimentRunner


def test_cli_help():
    out = subprocess.run(
        [sys.executable, "-m", "spuma.experiments", "--help"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={"PATH": "", "SPUMA_LOGGING__FILE": "false", "SPUMA_LOGGING__FORMAT": "text"},
    )
    assert out.returncode == 0, out.stderr
    assert all(flag in out.stdout for flag in ("--experiment", "--phase", "--limit", "--list"))


@pytest.mark.gpu
def test_real_qwen3_math_smoke(config, set_settings, tmp_path):
    """Phase 11: a real Qwen2.5-7B-Instruct run on one MATH problem (downloads the model and dataset)."""

    raw = config.to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["model"]["generation"]["max_new_tokens"] = 512
    set_settings(type(config).from_dict(raw))
    runner = ExperimentRunner()
    runner.run(phases=["generate", "score"], limit=1)
    feats = runner.store.read_json(f"{runner.scored_dir()}/features.json")
    assert len(feats) == 3 and all(len(r["features"]) == 8 for r in feats)
