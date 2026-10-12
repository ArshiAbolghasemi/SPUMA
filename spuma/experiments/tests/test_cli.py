import subprocess
import sys

import pytest

from spuma.config import PROJECT_ROOT


def test_list_variants(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "spuma.experiments", "--list"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={"PATH": "", "SPUMA_LOGGING__FILE": "false", "SPUMA_LOGGING__FORMAT": "text"},
    )
    assert out.returncode == 0, out.stderr
    assert "no_tools" in out.stderr and "both_tools" in out.stderr


@pytest.mark.gpu
def test_real_qwen3_math_smoke(config, tmp_path):
    """Phase 11: a real Qwen3-8B run on one MATH problem (downloads the model and dataset)."""
    from spuma.experiments.runner import ExperimentRunner

    raw = config.to_dict()
    raw["experiment"]["output_dir"] = str(tmp_path)
    raw["model"]["generation"]["max_new_tokens"] = 512
    cfg = type(config).from_dict(raw)
    runner = ExperimentRunner(cfg)
    runner.run(variants=["both_tools"], phases=["generate", "score"], limit=1)
    feats = runner.store.read_json(f"{runner.scored_dir('both_tools')}/features.json")
    assert len(feats) == 3 and all(len(r["features"]) == 8 for r in feats)
