"""LangGraph orchestration of the agents (`agents.workflow`), with uncertainty measured after every response.
One module per workflow, selected by `agents.workflow`: `feedbackloop` (feedback), `star` (star), `sequential`
(sequential); the shared machinery is in `base`.
"""

from collections.abc import Callable
from typing import Any

from spuma.agents.workflow.base import RunState, StepHook, Workflow, build_agents
from spuma.agents.workflow.feedbackloop import FeedbackWorkflow
from spuma.agents.workflow.sequential import MultiAgentWorkflow
from spuma.agents.workflow.star import StarState, StarWorkflow
from spuma.config import get_settings
from spuma.models.backend import ModelBackend

__all__ = [
    "FeedbackWorkflow",
    "MultiAgentWorkflow",
    "RunState",
    "StarState",
    "StarWorkflow",
    "StepHook",
    "Workflow",
    "build_agents",
    "build_workflow",
]


def build_workflow(
    backend_for: Callable[[str], ModelBackend], uq: Any = None, on_step: StepHook | None = None
) -> MultiAgentWorkflow | FeedbackWorkflow | StarWorkflow:
    """The `agents.workflow` graph; `uq` (a `StepUncertainty`) measures every response, `on_step` receives each
    measured step."""
    config = get_settings().agents
    agents = build_agents(backend_for)
    if config.workflow == "feedback":
        return FeedbackWorkflow(agents, config.max_rounds, config.completion_token, uq, on_step)
    if config.workflow == "star":
        return StarWorkflow(agents, config.completion_token, uq, on_step)
    return MultiAgentWorkflow(agents, uq, on_step, config.completion_token)
