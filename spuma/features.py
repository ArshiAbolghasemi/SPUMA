"""SPN inputs: the ordered feature vector Z (8 features: 4 metrics x 2 agents) per trajectory.

Tool-use metadata is kept in a separate dict so tool features can be added later as ablations
without changing Z.
"""

import logging
from typing import Any

from spuma.agents.trajectory import Trajectory
from spuma.config import get_settings
from spuma.uncertainty.aggregation import aggregate_agent

logger = logging.getLogger(__name__)


def extract_features(trajectory: Trajectory) -> dict[str, float]:
    """Z in `features.order`: each agent's step metrics aggregated per the `aggregation:` settings."""
    config = get_settings()
    metrics = config.uncertainty.enabled_metrics()
    values: dict[str, float] = {}
    for trace in trajectory.agents:
        for metric, v in aggregate_agent(trace, metrics).items():
            values[f"{trace.agent_id}_{metric}"] = v
    features = {name: values[name] for name in config.features.order}
    logger.info("features", extra={"problem_id": trajectory.problem_id, "features": features})
    return features


def tool_metadata(trajectory: Trajectory) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for trace in trajectory.agents:
        records = [r for s in trace.steps for r in s.tool_records]
        out[trace.agent_id] = {
            "effective_tools": trace.effective_tools,
            "steps": len(trace.steps),
            "tool_calls": len(records),
            "tool_successes": sum(r["status"] == "success" for r in records),
            "tool_errors": sum(r["status"] != "success" for r in records),
            "stop_reason": trace.stop_reason,
        }
    return out
