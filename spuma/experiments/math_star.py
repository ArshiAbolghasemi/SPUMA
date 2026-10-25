"""MATH with the fixed single-pass Star architecture: a Star coordinator, an Analyst and a Verifier, all
Qwen2.5-7B-Instruct (star_delegate -> analyst -> star_request_verification -> verifier -> star_finalize).

Everything else (splits, backends, generation, uncertainty, SPN, CHMM, evaluation) matches the feedback-loop
experiment; it is all in the settings file.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

EXPERIMENT = Experiment(
    name="math_ma_qwen2_5_7b_star",
    config_file=PROJECT_ROOT / "config" / "math_ma_qwen2_5_7b_star.yaml",
    description="MATH, Qwen2.5-7B-Instruct Star coordinator -> Analyst -> Verifier (fixed, single pass)",
)
