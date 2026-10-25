"""MATH with the fixed single-pass Sequential architecture: Analyst -> Verifier -> Finalizer, all
Qwen2.5-7B-Instruct, one call each. The Star roles without the coordinator.

Everything else (splits, backends, generation, uncertainty, SPN, CHMM, evaluation) matches the other MATH
experiments; it is all in the settings file.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

EXPERIMENT = Experiment(
    name="math_ma_qwen2_5_7b_sequential",
    config_file=PROJECT_ROOT / "config" / "sequential" / "math_ma_qwen2_5_7b_sequential.yaml",
    description="MATH, Qwen2.5-7B-Instruct Analyst -> Verifier -> Finalizer (fixed, single pass)",
)
