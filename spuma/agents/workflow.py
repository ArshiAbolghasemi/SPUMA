"""LangGraph orchestration of the agents (`agents.workflow`), with uncertainty measured after every response.

feedback (MATU / CAMEL-style role play of two agents: the first solves, the second asks and gives feedback):

    START -> mathematician -> uq_mathematician -> enthusiast -> uq_enthusiast -> (mathematician | END)

The loop ends when the second agent's response contains `agents.completion_token` (deterministic substring
match) or after `agents.max_rounds` rounds; one round is one response from each agent.

sequential: a chain of agents, one response each, each seeing the earlier agents' final responses.

    START -> agent_1 -> uq_agent_1 -> ... -> agent_k -> uq_agent_k -> END

A `uq_*` node measures the response just generated (`StepUncertainty`); its result goes to
`step_uncertainties`, never to `messages`, so uncertainty cannot change what the agents see or when the loop
stops. The graph only ever sees the problem text, never reference answers.
"""

import logging
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.trajectory import AgentTrace, StepRecord, Trajectory
from spuma.config import get_settings
from spuma.models.backend import ModelBackend

logger = logging.getLogger(__name__)

StepHook = Callable[[StepRecord], None]


class RunState(TypedDict):
    problem_id: str
    problem: str
    run_id: int
    round: int
    messages: Annotated[list[dict[str, Any]], operator.add]
    current_agent: str
    completed: bool
    termination_reason: str | None
    step_uncertainties: Annotated[list[dict[str, Any]], operator.add]
    generation_metadata: Annotated[list[StepRecord], operator.add]


def build_agents(backend_for: Callable[[str], ModelBackend]) -> list[LLMAgent]:
    """Enabled agents in settings order (the stable order that features use)."""
    return [LLMAgent(a.id, backend_for(a.id)) for a in get_settings().agents.enabled]


class _Workflow:
    """Shared machinery: generation and UQ nodes, running one independent run, assembling the trajectory."""

    def __init__(self, agents: list[LLMAgent], uq: Any = None, on_step: StepHook | None = None) -> None:
        self.agents, self.uq, self.on_step = agents, uq, on_step
        self.graph: Any = None
        self.recursion_limit = 25

    def view(self, agent: LLMAgent, state: RunState) -> list[BaseMessage]:
        raise NotImplementedError

    def _speak(self, agent: LLMAgent, state: RunState, round: int) -> dict[str, Any]:
        step_id = sum(s.agent_id == agent.id for s in state["generation_metadata"])
        record = agent.step(self.view(agent, state), state["problem_id"], step_id, state["run_id"], round)
        return {
            "messages": [{"agent_id": agent.id, "round": round, "content": record.content}],
            "generation_metadata": [record],
            "current_agent": agent.id,
        }

    def _measure(self, marker: str | None) -> Callable[[RunState], dict[str, Any]]:
        def node(state: RunState) -> dict[str, Any]:
            record = state["generation_metadata"][-1]
            if self.uq is None:
                return {}
            measured = record.model_copy(deep=True)
            result = self.uq.measure(measured, state["problem"], marker)
            entry = {
                "agent_id": record.agent_id,
                "round": record.round,
                "step_id": record.step_id,
                "metrics": result["metrics"],
                "uq_details": {**measured.uq_details, **result["details"]},
            }
            if self.on_step:
                self.on_step(_merge(record, entry))
            return {"step_uncertainties": [entry]}

        return node

    def _initial(self, problem_id: str, problem: str, run_id: int) -> RunState:
        return {
            "problem_id": problem_id,
            "problem": problem,
            "run_id": run_id,
            "round": 0,
            "messages": [],
            "current_agent": "",
            "completed": False,
            "termination_reason": None,
            "step_uncertainties": [],
            "generation_metadata": [],
        }

    def run(self, problem_id: str, problem: str, run_id: int = 0) -> Trajectory:
        """One independent run from a fresh state. An exception ends the run with `termination_reason: error`
        and keeps the steps finished before it."""
        state = initial = self._initial(problem_id, problem, run_id)
        error = None
        try:
            for latest in self.graph.stream(initial, {"recursion_limit": self.recursion_limit}, stream_mode="values"):
                state = latest
        except Exception as e:  # noqa: BLE001 - recorded on the run; the other runs go on
            logger.exception("run failed", extra={"problem_id": problem_id, "run_id": run_id})
            error = f"{type(e).__name__}: {e}"
        reason = "error" if error else self.termination_reason(state)
        trajectory = self._trajectory(state, reason, error)
        logger.info(
            "run complete",
            extra={"problem_id": problem_id, "run_id": run_id, "rounds": trajectory.rounds, "termination": reason},
        )
        return trajectory

    def termination_reason(self, state: RunState) -> str:
        raise NotImplementedError

    def final_answer(self, traces: list[AgentTrace]) -> str:
        return traces[-1].final_content

    def _trajectory(self, state: RunState, reason: str, error: str | None) -> Trajectory:
        uq = {(u["agent_id"], u["step_id"]): u for u in state["step_uncertainties"]}
        traces = []
        for agent in self.agents:
            steps = [
                _merge(s, uq.get((s.agent_id, s.step_id)))
                for s in state["generation_metadata"]
                if s.agent_id == agent.id
            ]
            final = steps[-1].content if steps else ""
            traces.append(
                AgentTrace(
                    agent_id=agent.id, role=agent.config.role, model=agent.model_name, steps=steps, final_content=final
                )
            )
        return Trajectory(
            problem_id=state["problem_id"],
            problem=state["problem"],
            agents=traces,
            final_answer_text=self.final_answer(traces),
            run_id=state["run_id"],
            rounds=state["round"],
            termination_reason=reason,
            error=error,
            messages=state["messages"],
        )


def _merge(record: StepRecord, uq: dict[str, Any] | None) -> StepRecord:
    if uq is None:
        return record
    return record.model_copy(update={"metrics": uq["metrics"], "uq_details": uq["uq_details"]})


class MultiAgentWorkflow(_Workflow):
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


class FeedbackWorkflow(_Workflow):
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


def build_workflow(
    backend_for: Callable[[str], ModelBackend], uq: Any = None, on_step: StepHook | None = None
) -> MultiAgentWorkflow | FeedbackWorkflow:
    """The `agents.workflow` graph; `uq` (a `StepUncertainty`) measures every response, `on_step` receives each
    measured step."""
    config = get_settings().agents
    agents = build_agents(backend_for)
    if config.workflow == "feedback":
        return FeedbackWorkflow(agents, config.max_rounds, config.completion_token, uq, on_step)
    return MultiAgentWorkflow(agents, uq, on_step)
