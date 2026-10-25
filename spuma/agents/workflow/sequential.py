"""Sequential workflow (`agents.workflow: sequential`): a chain of agents, one response each, each seeing the
earlier agents' final responses.

    START -> agent_1 -> uq_agent_1 -> ... -> agent_k -> uq_agent_k -> END
"""

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.workflow.base import RunState, StepHook, Workflow


class MultiAgentWorkflow(Workflow):
    """Sequential chain, one response per agent."""

    def __init__(self, agents: list[LLMAgent], uq: Any = None, on_step: StepHook | None = None) -> None:
        super().__init__(agents, uq, on_step)
        graph = StateGraph(RunState)
        previous = START
        for agent in agents:
            graph.add_node(agent.id, lambda state, a=agent: self._speak(a, state, 1) | {"round": 1})
            graph.add_node(f"uq_{agent.id}", self._measure(None))
            graph.add_edge(previous, agent.id)
            graph.add_edge(agent.id, f"uq_{agent.id}")
            previous = f"uq_{agent.id}"
        graph.add_edge(previous, END)
        self.graph = graph.compile()
        self.recursion_limit = 2 * len(agents) + 5

    def view(self, agent: LLMAgent, state: RunState) -> list[BaseMessage]:
        previous = [m["content"] for m in state["messages"]]
        return agent.initial_messages(state["problem"], previous)

    def termination_reason(self, state: RunState) -> str:
        return "completed"
