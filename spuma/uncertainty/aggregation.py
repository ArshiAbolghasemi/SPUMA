"""Per-agent aggregation of step metrics: RMS for PE/SE/NLL, arithmetic mean for token length (configurable)."""

import math
from collections.abc import Callable, Sequence

from spuma.agents.trajectory import AgentTrace
from spuma.config import get_settings

AggregationStrategy = Callable[[Sequence[float]], float]


def rms(values: Sequence[float]) -> float:
    """sqrt((1/T) Σ u_t²)"""
    _nonempty(values)
    return math.sqrt(sum(v * v for v in values) / len(values))


def mean(values: Sequence[float]) -> float:
    _nonempty(values)
    return sum(values) / len(values)


def _nonempty(values: Sequence[float]) -> None:
    if not values:
        raise ValueError("cannot aggregate an agent with no steps")


AGGREGATORS: dict[str, AggregationStrategy] = {"rms": rms, "mean": mean, "max": lambda v: (_nonempty(v), max(v))[1]}


def register_aggregator(name: str, fn: AggregationStrategy) -> None:
    AGGREGATORS[name] = fn


def aggregate_agent(trace: AgentTrace, metrics: Sequence[str]) -> dict[str, float]:
    """Each metric over the agent's steps, with the strategy named in the `aggregation:` settings."""
    config = get_settings().aggregation
    out = {}
    for m in metrics:
        name = config.for_metric(m)
        if name not in AGGREGATORS:
            raise ValueError(f"unknown aggregation {name!r} for metric {m!r}; known: {sorted(AGGREGATORS)}")
        out[m] = AGGREGATORS[name]([s.metrics[m] for s in trace.steps])
    return out
