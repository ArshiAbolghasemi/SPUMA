"""Persisted trajectory records.

A *step* is one model call: one assistant message generated from a frozen context. Everything
needed to replay it is kept (rendered messages, exact prompt token ids, generation parameters, seed).
Each agent makes exactly one step.
"""

from typing import Any

from pydantic import BaseModel, Field


class StepRecord(BaseModel):
    problem_id: str
    agent_id: str
    step_id: int
    input_messages: list[dict[str, Any]]  # chat-template dicts, the exact conversation rendered
    prompt_ids: list[int]  # exact model input
    generation_params: dict[str, Any]
    seed: int
    token_ids: list[int]  # model-generated tokens only
    text: str
    # Raw-model scores of token_ids; `scored_by` names the engine that produced them (None: not scored yet).
    token_logprobs: list[float] | None = None
    token_entropies: list[float] | None = None
    scored_by: str | None = None
    generation_meta: dict[str, Any] = Field(default_factory=dict)  # backend, finish reason, latency, ...
    content: str = ""
    reasoning: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)  # filled by spuma.uncertainty
    uq_details: dict[str, Any] = Field(default_factory=dict)  # semantic-entropy samples, clusters, policy


class AgentTrace(BaseModel):
    agent_id: str
    role: str
    model: str
    steps: list[StepRecord] = Field(default_factory=list)
    final_content: str = ""


class Trajectory(BaseModel):
    problem_id: str
    problem: str
    agents: list[AgentTrace]
    final_answer_text: str  # the last agent's (Reviewer's) final message

    def agent(self, agent_id: str) -> AgentTrace:
        return next(a for a in self.agents if a.agent_id == agent_id)
