"""Sequential workflow (`agents.workflow: sequential`), fixed and single pass: three agents (solver, verifier,
finalizer: the enabled agents, in that order), one response each.

    START -> solver -> uq_solver -> verifier -> uq_verifier -> finalizer -> uq_finalizer -> END

Exactly three model calls, no loop and no routing decision. Each agent gets its system prompt and one user
message, its `user_template` filled with `{problem}`, `{previous}` (the earlier agents' responses, in order,
joined by a blank line) and `{<agent id>}` (each earlier agent's response, e.g. `{solver}`). The finalizer's
response is graded. `TASK_DONE` is stripped before scoring on every node; it is expected only from the finalizer
and never changes the control flow.
"""

from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.workflow.base import RunState, StepHook, Workflow


class MultiAgentWorkflow(Workflow):
    """Sequential chain, one response per agent."""

    def __init__(
        self,
        agents: list[LLMAgent],
        uq: Any = None,
        on_step: StepHook | None = None,
        completion_token: str = "TASK_DONE",
    ) -> None:
        super().__init__(agents, uq, on_step)
        self.completion_token = completion_token
        graph = StateGraph(RunState)
        previous = START
        for agent in agents:
            graph.add_node(agent.id, lambda state, a=agent: self._speak(a, state, 1) | {"round": 1})
            graph.add_node(f"uq_{agent.id}", self._measure(completion_token))
            graph.add_edge(previous, agent.id)
            graph.add_edge(agent.id, f"uq_{agent.id}")
            previous = f"uq_{agent.id}"
        graph.add_edge(previous, END)
        self.graph = graph.compile()
        self.recursion_limit = 2 * len(agents) + 5

    def view(self, agent: LLMAgent, state: RunState) -> list[BaseMessage]:
        earlier = {m["agent_id"]: m["content"] for m in state["messages"]}
        user = agent.config.user_template.format(
            problem=state["problem"], previous="\n\n".join(earlier.values()), **earlier
        )
        return [SystemMessage(agent.config.system_prompt), HumanMessage(user)]

    def termination_reason(self, state: RunState) -> str:
        final = state["messages"][-1]["content"] if state["messages"] else ""
        return "task_done" if self.completion_token in final else "no_completion_token"

    def metadata(self, state: RunState) -> dict[str, Any]:
        early = [m["agent_id"] for m in state["messages"][:-1] if self.completion_token in m["content"]]
        return {"workflow": "sequential", "llm_calls": len(state["messages"]), "early_completion_token": early}
