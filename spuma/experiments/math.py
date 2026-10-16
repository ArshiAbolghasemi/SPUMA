"""MATH (EleutherAI/hendrycks_math), MATU-style: two Qwen2.5-7B-Instruct agents in a feedback loop
(Master Mathematician <-> Curious Math Enthusiast), several independent runs per problem.

Everything else (splits, backends, prompts, uncertainty, SPN, evaluation) is in its settings file.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

EXPERIMENT = Experiment(
    name="math",
    config_file=PROJECT_ROOT / "config" / "config-qwen-math.yaml",
    description="MATH, Qwen2.5-7B-Instruct Mathematician <-> Enthusiast feedback loop",
)
