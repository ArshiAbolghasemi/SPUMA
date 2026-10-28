"""Registered experiments, by name: every (dataset, architecture) pair, each with its settings file
`config/<architecture>/<dataset>_ma_qwen2_5_7b_<architecture>.yaml`. A new dataset or architecture adds its
settings files and one entry here; an experiment that needs code of its own subclasses `Experiment`.

MATH comes first: its trained models are the frozen sources (`chmm.frozen_from`) of the other datasets.
"""

from spuma.config import PROJECT_ROOT
from spuma.experiments.base import Experiment

ARCHITECTURES = ("feedbackloop", "star", "sequential")
DATASETS = {
    "math": "MATH competition mathematics",
    "mmlu": "MMLU four-choice academic questions",
    "musique": "MuSiQue-Ans evidence-grounded multi-hop QA",
    "morehopqa": "MoreHopQA multi-hop QA with a reasoning step on the last hop",
}

EXPERIMENTS: dict[str, Experiment] = {
    f"{dataset}_ma_qwen2_5_7b_{architecture}": Experiment(
        name=f"{dataset}_ma_qwen2_5_7b_{architecture}",
        config_file=PROJECT_ROOT / "config" / architecture / f"{dataset}_ma_qwen2_5_7b_{architecture}.yaml",
        description=f"{description}, Qwen2.5-7B-Instruct {architecture}",
        architecture=architecture,
        dataset=dataset,
    )
    for dataset, description in DATASETS.items()
    for architecture in ARCHITECTURES
}
