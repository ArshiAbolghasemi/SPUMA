"""Feedback-loop workflow (`agents.workflow: feedback`): MATU / CAMEL-style role play of two agents, the first
solves, the second asks and gives feedback:

    START -> mathematician -> uq_mathematician -> enthusiast -> uq_enthusiast -> (mathematician | END)

The loop ends when the second agent's response contains `agents.completion_token` (deterministic substring
match) or after `agents.max_rounds` rounds; one round is one response from each agent.
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.trajectory import AgentTrace
from spuma.agents.workflow.base import RunState, StepHook, Workflow


class FeedbackWorkflow(Workflow):
    """Two-agent feedback loop ending at the completion token or after `max_rounds` rounds."""

    def __init__(
        self,
        agents: list[LLMAgent],
        max_rounds: int,
        completion_token: str,
        uq: Any = None,
        on_step: StepHook | None = None,
    ) -> None:
        super().__init__(agents, uq, on_step)
        self.solver, self.critic = agents
        self.max_rounds, self.completion_token = max_rounds, completion_token
        graph = StateGraph(RunState)
        graph.add_node(self.solver.id, self._solver_node)
        graph.add_node(f"uq_{self.solver.id}", self._measure(None))
        graph.add_node(self.critic.id, self._critic_node)
        graph.add_node(f"uq_{self.critic.id}", self._measure(completion_token))
        graph.add_edge(START, self.solver.id)
        graph.add_edge(self.solver.id, f"uq_{self.solver.id}")
        graph.add_edge(f"uq_{self.solver.id}", self.critic.id)
        graph.add_edge(self.critic.id, f"uq_{self.critic.id}")
        graph.add_conditional_edges(f"uq_{self.critic.id}", self._route, [self.solver.id, END])
        self.graph = graph.compile()
        self.recursion_limit = 4 * max_rounds + 5

    def view(self, agent: LLMAgent, state: RunState) -> list[BaseMessage]:
        """The whole conversation as `agent` sees it, its own responses as assistant messages.

        Solver: its user template around the problem, then each critic response verbatim as a user message.
        Critic: each solver response inside its user template (problem + latest solver response) as a user
        message."""
        problem = state["problem"]
        messages: list[BaseMessage] = [SystemMessage(agent.config.system_prompt)]
        if agent is self.solver:
            messages.append(HumanMessage(agent.config.user_template.format(problem=problem, previous="")))
        for turn in state["messages"]:
            if turn["agent_id"] == agent.id:
                messages.append(AIMessage(turn["content"]))
            elif agent is self.critic:
                messages.append(
                    HumanMessage(agent.config.user_template.format(problem=problem, previous=turn["content"]))
                )
            else:
                messages.append(HumanMessage(turn["content"]))
        return messages

    def _solver_node(self, state: RunState) -> dict[str, Any]:
        round = state["round"] + 1
        return self._speak(self.solver, state, round) | {"round": round}

    def _critic_node(self, state: RunState) -> dict[str, Any]:
        update = self._speak(self.critic, state, state["round"])
        done = self.completion_token in update["generation_metadata"][0].text
        return update | {"completed": done, "termination_reason": "task_done" if done else None}

    def _route(self, state: RunState) -> str:
        return END if state["completed"] or state["round"] >= self.max_rounds else self.solver.id

    def termination_reason(self, state: RunState) -> str:
        return "task_done" if state["completed"] else "max_rounds"

    def final_answer(self, traces: list[AgentTrace]) -> str:
        return traces[0].final_content
