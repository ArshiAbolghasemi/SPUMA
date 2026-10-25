"""MATH (EleutherAI/hendrycks_math), MATU-style: two Qwen2.5-7B-Instruct agents in a feedback loop
(Master Mathematician <-> Curious Math Enthusiast).

Everything else (splits, backends, prompts, uncertainty, SPN, evaluation) is in its settings file.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

EXPERIMENT = Experiment(
    name="math_ma_qwen2_5_7b_feedbackloop",
    config_file=PROJECT_ROOT / "config" / "feedbackloop" / "math_ma_qwen2_5_7b_feedbackloop.yaml",
    description="MATH, Qwen2.5-7B-Instruct Mathematician <-> Enthusiast feedback loop",
)
