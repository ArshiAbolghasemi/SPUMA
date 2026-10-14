"""MATH (EleutherAI/hendrycks_math) with two Qwen3-8B agents: Solver -> Reviewer, each with its SymPy tools.

Everything else (splits, backends, tool permissions, uncertainty, SPN, evaluation) is in its settings file.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

EXPERIMENT = Experiment(
    name="math",
    config_file=PROJECT_ROOT / "config" / "config-qwen-math.yaml",
    description="MATH, Qwen3-8B Solver -> Reviewer",
)
