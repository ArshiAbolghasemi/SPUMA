"""LangGraph orchestration: a sequential chain of agents, each with its own tool-calling loop.

    START -> solver <-> solver_tools -> reviewer <-> reviewer_tools -> END

An agent node makes exactly one model call (one step). If that step requested tools and the agent
has steps left, its tools node executes them through the permission-enforcing `ToolExecutor` and
loops back; otherwise control passes to the next agent, which receives the problem plus the earlier
agents' final messages. The graph only ever sees the problem text, never reference answers.
"""

import logging
from collections.abc import Callable
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from spuma.agents.agent import LLMAgent
from spuma.agents.trajectory import AgentTrace, Trajectory
from spuma.config import get_settings
from spuma.models.backend import ModelBackend
from spuma.tools.registry import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


class State(TypedDict):
    problem_id: str
    problem: str
    traces: dict[str, AgentTrace]
    messages: dict[str, list[BaseMessage]]


def build_agents(
    registry: ToolRegistry, backend_for: Callable[[str], ModelBackend], variant: str | None = None
) -> list[LLMAgent]:
    """Enabled agents in settings order (the stable order that features use)."""
    return [LLMAgent(a.id, backend_for(a.id), registry, variant) for a in get_settings(variant).agents.enabled]


class MultiAgentWorkflow:
    def __init__(self, agents: list[LLMAgent], executor: ToolExecutor) -> None:
        self.agents = agents
        self.executor = executor
        self.graph = self._build()
        # Each step is at most two graph super-steps (agent + tools), plus slack.
        self.recursion_limit = sum(2 * a.max_steps for a in agents) + 10

    def _build(self) -> Any:
        graph = StateGraph(State)
        for i, agent in enumerate(self.agents):
            nxt = self.agents[i + 1].id if i + 1 < len(self.agents) else END
            graph.add_node(agent.id, self._agent_node(agent, [a.id for a in self.agents[:i]]))
            graph.add_node(f"{agent.id}_tools", self._tools_node(agent))
            graph.add_conditional_edges(agent.id, self._router(agent, nxt), [f"{agent.id}_tools", nxt])
            graph.add_edge(f"{agent.id}_tools", agent.id)
        graph.add_edge(START, self.agents[0].id)
        return graph.compile()

    def _agent_node(self, agent: LLMAgent, previous_ids: list[str]) -> Callable[[State], dict[str, Any]]:
        def node(state: State) -> dict[str, Any]:
            traces, messages = dict(state["traces"]), dict(state["messages"])
            if agent.id not in traces:
                previous = [traces[p].final_content for p in previous_ids]
                messages[agent.id] = agent.initial_messages(state["problem"], previous)
                traces[agent.id] = AgentTrace(
                    agent_id=agent.id,
                    role=agent.config.role,
                    model=agent.model_name,
                    allowed_tools=agent.allowed_tools,
                    effective_tools=agent.effective_tools,
                )
            trace = traces[agent.id].model_copy()
            record, ai = agent.step(messages[agent.id], state["problem_id"], len(trace.steps))
            trace.steps = [*trace.steps, record]
            messages[agent.id] = [*messages[agent.id], ai]
            if not record.has_tool_calls:
                trace.final_content, trace.stop_reason = record.content, "final_message"
            elif len(trace.steps) >= agent.max_steps:
                trace.final_content, trace.stop_reason = record.content, "max_steps"
            traces[agent.id] = trace
            return {"traces": traces, "messages": messages}

        return node

    def _tools_node(self, agent: LLMAgent) -> Callable[[State], dict[str, Any]]:
        def node(state: State) -> dict[str, Any]:
            traces, messages = dict(state["traces"]), dict(state["messages"])
            trace = traces[agent.id].model_copy()
            step = trace.steps[-1].model_copy()
            records, observations = [], []
            for call in [*step.tool_calls, *step.invalid_tool_calls]:
                record = self.executor.execute(
                    call,
                    problem_id=state["problem_id"],
                    agent_id=agent.id,
                    step_id=step.step_id,
                    allowed_tools=agent.allowed_tools,
                    calls_used=trace.tool_calls_used,
                    max_calls=agent.max_tool_calls,
                    message_tokens=len(step.token_ids),
                )
                trace.tool_calls_used += 1  # every attempt counts against the budget
                records.append(record.to_dict())
                observations.append(
                    ToolMessage(record.observation, tool_call_id=record.call_id, name=record.tool or "")
                )
            step.tool_records = records
            trace.steps = [*trace.steps[:-1], step]
            traces[agent.id] = trace
            messages[agent.id] = [*messages[agent.id], *observations]
            return {"traces": traces, "messages": messages}

        return node

    @staticmethod
    def _router(agent: LLMAgent, nxt: str) -> Callable[[State], str]:
        def route(state: State) -> str:
            return nxt if state["traces"][agent.id].stop_reason else f"{agent.id}_tools"

        return route

    def run(self, problem_id: str, problem: str) -> Trajectory:
        state = self.graph.invoke(
            {"problem_id": problem_id, "problem": problem, "traces": {}, "messages": {}},
            {"recursion_limit": self.recursion_limit},
        )
        agents = [state["traces"][a.id] for a in self.agents]
        logger.info(
            "trajectory complete",
            extra={
                "problem_id": problem_id,
                "steps": {a.agent_id: len(a.steps) for a in agents},
                "stop": {a.agent_id: a.stop_reason for a in agents},
            },
        )
        return Trajectory(
            problem_id=problem_id, problem=problem, agents=agents, final_answer_text=agents[-1].final_content
        )


def build_workflow(
    registry: ToolRegistry,
    backend_for: Callable[[str], ModelBackend],
    executor: ToolExecutor,
    variant: str | None = None,
) -> MultiAgentWorkflow:
    """The `agents.workflow` graph; `variant` selects an experiment variant's allowed_tools."""
    workflow = get_settings(variant).agents.workflow
    if workflow != "sequential":
        raise ValueError(f"unsupported workflow {workflow!r}")
    return MultiAgentWorkflow(build_agents(registry, backend_for, variant), executor)
