"""Star workflow (`agents.workflow: star`), fixed and single pass: a coordinator LLM, an analyst and a verifier
(the enabled agents, in that order).

    START -> star_delegate -> analyst -> star_request_verification -> verifier -> star_finalize -> END

(each node followed by its `uq_*` node). Exactly five model calls, no loop and no routing decision: the edges are
fixed and the run always ends after `star_finalize`. The coordinator's three calls share its system prompt; each
gets the phase's template (`templates.delegation` / `verification_request` / `final_synthesis`) and the phase is
in the state. The coordinator sees its accumulated conversation; the analyst sees the problem and the
delegation; the verifier sees the problem, the analyst's solution and the verification request. A step's
`round` is its node's position (1-5), so each call has its own seed. `TASK_DONE` is stripped before scoring on
every node; it is expected only in the final synthesis and never changes the control flow.
"""

import logging
import operator
from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.config import STAR_PHASES
from spuma.agents.trajectory import AgentTrace
from spuma.agents.workflow.base import RunState, StepHook, Workflow

logger = logging.getLogger(__name__)


class StarState(RunState):
    phase: str
    analyst_solution: str
    verifier_assessment: str
    final_answer: str
    usage: Annotated[list[dict[str, Any]], operator.add]


class StarWorkflow(Workflow):
    """Fixed single-pass star: coordinator -> analyst -> coordinator -> verifier -> coordinator -> END."""

    NODES = (
        ("star_delegate", "coordinator", "delegation"),
        ("analyst", "analyst", None),
        ("star_request_verification", "coordinator", "verification_request"),
        ("verifier", "verifier", None),
        ("star_finalize", "coordinator", "final_synthesis"),
    )

    def __init__(
        self, agents: list[LLMAgent], completion_token: str, uq: Any = None, on_step: StepHook | None = None
    ) -> None:
        super().__init__(agents, uq, on_step)
        self.coordinator, self.analyst, self.verifier = agents
        self.completion_token = completion_token
        graph = StateGraph(StarState)
        previous = START
        for position, (node, role, phase) in enumerate(self.NODES, start=1):
            graph.add_node(node, lambda state, r=role, p=phase, n=node, i=position: self._call(state, r, p, n, i))
            graph.add_node(f"uq_{node}", self._measure(completion_token))
            graph.add_edge(previous, node)
            graph.add_edge(node, f"uq_{node}")
            previous = f"uq_{node}"
        graph.add_edge(previous, END)
        self.graph = graph.compile()
        self.recursion_limit = 2 * len(self.NODES) + 5

    def _initial(self, problem_id: str, problem: str) -> StarState:
        return super()._initial(problem_id, problem) | {
            "round": 1,
            "phase": "",
            "analyst_solution": "",
            "verifier_assessment": "",
            "final_answer": "",
            "usage": [],
        }

    def _coordinator_view(self, state: StarState, phase: str) -> list[BaseMessage]:
        """System prompt, then per phase so far: the phase's template (user) and the coordinator's reply."""
        config = self.coordinator.config
        context = {
            "problem": state["problem"],
            "solution": state["analyst_solution"],
            "assessment": state["verifier_assessment"],
        }
        own = [m["content"] for m in state["messages"] if m["agent_id"] == self.coordinator.id]
        messages: list[BaseMessage] = [SystemMessage(config.system_prompt)]
        for done, reply in zip(STAR_PHASES, own, strict=False):
            messages += [HumanMessage(config.templates[done].format(**context)), AIMessage(reply)]
        return [*messages, HumanMessage(config.templates[phase].format(**context))]

    def _request(self, state: StarState) -> str:
        """The coordinator's latest message (the delegation for the analyst, the request for the verifier)."""
        return next(m["content"] for m in reversed(state["messages"]) if m["agent_id"] == self.coordinator.id)

    def view(self, agent: LLMAgent, state: StarState) -> list[BaseMessage]:
        context = {"problem": state["problem"], "request": self._request(state), "solution": state["analyst_solution"]}
        return [SystemMessage(agent.config.system_prompt), HumanMessage(agent.config.user_template.format(**context))]

    def _call(self, state: StarState, role: str, phase: str | None, node: str, position: int) -> dict[str, Any]:
        agent = {"coordinator": self.coordinator, "analyst": self.analyst, "verifier": self.verifier}[role]
        messages = self._coordinator_view(state, phase) if phase else self.view(agent, state)
        update = self._speak(agent, state, position, messages)
        record = update["generation_metadata"][0]
        update["messages"][0] |= {"node": node, "phase": phase}
        update["usage"] = [
            {
                "node": node,
                "agent_id": agent.id,
                "phase": phase,
                "prompt_tokens": len(record.prompt_ids),
                "completion_tokens": len(record.token_ids),
            }
        ]
        if self.completion_token in record.text and node != "star_finalize":
            logger.warning("completion token before the final synthesis (ignored)", extra={"node": node})
        if role == "analyst":
            update["analyst_solution"] = record.content
        elif role == "verifier":
            update["verifier_assessment"] = record.content
        if phase:
            update["phase"] = phase
        if node == "star_finalize":
            done = self.completion_token in record.text
            update |= {
                "final_answer": record.content,
                "completed": done,
                "termination_reason": "task_done" if done else "no_completion_token",
            }
        return update

    def termination_reason(self, state: StarState) -> str:
        return state["termination_reason"] or "incomplete"

    def final_answer(self, traces: list[AgentTrace]) -> str:
        return traces[0].final_content

    def metadata(self, state: StarState) -> dict[str, Any]:
        early = [m["node"] for m in state["messages"][:-1] if self.completion_token in m["content"]]
        return {
            "workflow": "star",
            "phase": state["phase"],
            "analyst_solution": state["analyst_solution"],
            "verifier_assessment": state["verifier_assessment"],
            "final_answer": state["final_answer"],
            "completed": state["completed"],
            "early_completion_token": early,
            "llm_calls": len(state["usage"]),
            "usage": state["usage"],
        }
