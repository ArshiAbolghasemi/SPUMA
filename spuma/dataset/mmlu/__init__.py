"""MMLU (Hendrycks et al., 2021): four-choice academic and professional questions, `cais/mmlu`."""

from spuma.dataset.mmlu.loader import MMLUAdapter, MMLUDataset, MMLUGrader

__all__ = ["MMLUAdapter", "MMLUDataset", "MMLUGrader"]
