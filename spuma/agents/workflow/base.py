"""Shared workflow machinery: graph state, generation and UQ nodes, running a problem, assembling the trajectory.

A `uq_*` node measures the response just generated (`StepUncertainty`); its result goes to
`step_uncertainties`, never to `messages`, so uncertainty cannot change what the agents see or when the loop
stops. The graph only ever sees the problem text, never reference answers.
"""

import logging
import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from spuma.agents.agent import LLMAgent
from spuma.agents.trajectory import AgentTrace, StepRecord, Trajectory
from spuma.config import get_settings
from spuma.models.backend import ModelBackend

logger = logging.getLogger(__name__)

StepHook = Callable[[StepRecord], None]


class RunState(TypedDict):
    problem_id: str
    problem: str
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


class Workflow:
    """Shared machinery: generation and UQ nodes, running a problem from a fresh state, assembling the trajectory."""

    def __init__(self, agents: list[LLMAgent], uq: Any = None, on_step: StepHook | None = None) -> None:
        self.agents, self.uq, self.on_step = agents, uq, on_step
        self.graph: Any = None
        self.recursion_limit = 25

    def view(self, agent: LLMAgent, state: RunState) -> list[BaseMessage]:
        raise NotImplementedError

    def _speak(
        self, agent: LLMAgent, state: RunState, round: int, messages: list[BaseMessage] | None = None
    ) -> dict[str, Any]:
        step_id = sum(s.agent_id == agent.id for s in state["generation_metadata"])
        messages = self.view(agent, state) if messages is None else messages
        record = agent.step(messages, state["problem_id"], step_id, round)
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

    def _initial(self, problem_id: str, problem: str) -> RunState:
        return {
            "problem_id": problem_id,
            "problem": problem,
            "round": 0,
            "messages": [],
            "current_agent": "",
            "completed": False,
            "termination_reason": None,
            "step_uncertainties": [],
            "generation_metadata": [],
        }

    def run(self, problem_id: str, problem: str) -> Trajectory:
        """The problem from a fresh state. An exception ends the run with `termination_reason: error`
        and keeps the steps finished before it."""
        state = initial = self._initial(problem_id, problem)
        error = None
        try:
            for latest in self.graph.stream(initial, {"recursion_limit": self.recursion_limit}, stream_mode="values"):
                state = latest
        except Exception as e:  # noqa: BLE001 - recorded on the run; the other runs go on
            logger.exception("run failed", extra={"problem_id": problem_id})
            error = f"{type(e).__name__}: {e}"
        reason = "error" if error else self.termination_reason(state)
        trajectory = self._trajectory(state, reason, error)
        logger.info(
            "run complete",
            extra={"problem_id": problem_id, "rounds": trajectory.rounds, "termination": reason},
        )
        return trajectory

    def termination_reason(self, state: RunState) -> str:
        raise NotImplementedError

    def final_answer(self, traces: list[AgentTrace]) -> str:
        return traces[-1].final_content

    def metadata(self, state: RunState) -> dict[str, Any]:
        return {}

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
            rounds=state["round"],
            termination_reason=reason,
            error=error,
            messages=state["messages"],
            metadata=self.metadata(state),
        )


def _merge(record: StepRecord, uq: dict[str, Any] | None) -> StepRecord:
    if uq is None:
        return record
    return record.model_copy(update={"metrics": uq["metrics"], "uq_details": uq["uq_details"]})
