"""Registered experiments, by name. A new experiment adds a module with an `EXPERIMENT` and one entry here."""

from spuma.experiments import math
from spuma.experiments.base import Experiment

EXPERIMENTS: dict[str, Experiment] = {e.name: e for e in (math.EXPERIMENT,)}
