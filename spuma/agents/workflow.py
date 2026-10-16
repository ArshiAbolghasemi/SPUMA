"""LangGraph orchestration: a sequential chain of agents, one model call (one step) each.

    START -> solver -> reviewer -> END

Each agent receives the problem plus the earlier agents' final messages. The graph only ever sees the
problem text, never reference answers.
"""

import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.trajectory import AgentTrace, Trajectory
from spuma.config import get_settings
from spuma.models.backend import ModelBackend

logger = logging.getLogger(__name__)


class State(TypedDict):
    problem_id: str
    problem: str
    traces: dict[str, AgentTrace]


def build_agents(backend_for: Callable[[str], ModelBackend]) -> list[LLMAgent]:
    """Enabled agents in settings order (the stable order that features use)."""
    return [LLMAgent(a.id, backend_for(a.id)) for a in get_settings().agents.enabled]


class MultiAgentWorkflow:
    def __init__(self, agents: list[LLMAgent]) -> None:
        self.agents = agents
        self.graph = self._build()

    def _build(self) -> Any:
        graph = StateGraph(State)
        previous = START
        for i, agent in enumerate(self.agents):
            graph.add_node(agent.id, self._agent_node(agent, [a.id for a in self.agents[:i]]))
            graph.add_edge(previous, agent.id)
            previous = agent.id
        graph.add_edge(previous, END)
        return graph.compile()

    def _agent_node(self, agent: LLMAgent, previous_ids: list[str]) -> Callable[[State], dict[str, Any]]:
        def node(state: State) -> dict[str, Any]:
            traces = dict(state["traces"])
            previous = [traces[p].final_content for p in previous_ids]
            messages = agent.initial_messages(state["problem"], previous)
            record = agent.step(messages, state["problem_id"], 0)
            traces[agent.id] = AgentTrace(
                agent_id=agent.id,
                role=agent.config.role,
                model=agent.model_name,
                steps=[record],
                final_content=record.content,
            )
            return {"traces": traces}

        return node

    def run(self, problem_id: str, problem: str) -> Trajectory:
        state = self.graph.invoke({"problem_id": problem_id, "problem": problem, "traces": {}})
        agents = [state["traces"][a.id] for a in self.agents]
        logger.info("trajectory complete", extra={"problem_id": problem_id, "agents": [a.agent_id for a in agents]})
        return Trajectory(
            problem_id=problem_id, problem=problem, agents=agents, final_answer_text=agents[-1].final_content
        )


def build_workflow(backend_for: Callable[[str], ModelBackend]) -> MultiAgentWorkflow:
    """The `agents.workflow` graph."""
    workflow = get_settings().agents.workflow
    if workflow != "sequential":
        raise ValueError(f"unsupported workflow {workflow!r}")
    return MultiAgentWorkflow(build_agents(backend_for))
