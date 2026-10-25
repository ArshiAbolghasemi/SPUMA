"""MuSiQue (Trivedi et al., 2022): multi-hop questions over 20 context paragraphs (supporting + distractors)."""

from spuma.dataset.musique.loader import MuSiQueAdapter, MuSiQueDataset, MuSiQueGrader

__all__ = ["MuSiQueAdapter", "MuSiQueDataset", "MuSiQueGrader"]
