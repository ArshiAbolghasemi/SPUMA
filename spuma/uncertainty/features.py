"""SPN inputs: the ordered feature vector Z (8 features: 4 metrics x 2 agents) per trajectory."""

import logging

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
