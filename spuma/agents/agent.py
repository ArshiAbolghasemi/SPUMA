"""An agent = role prompt + model backend + generation settings + effective tools + limits."""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from spuma.agents.trajectory import StepRecord
from spuma.common.config import to_plain
from spuma.common.utils import derive_seed
from spuma.config import get_settings
from spuma.models.backend import GenerationRequest, ModelBackend, generation_logprob_mode, require
from spuma.models.chat import RAW_TEXT_KEY, to_chat_dicts
from spuma.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Agent(Protocol):
    id: str

    def initial_messages(self, problem: str, previous: list[str]) -> list[BaseMessage]: ...

    def step(self, messages: Sequence[BaseMessage], problem_id: str, step_id: int) -> tuple[StepRecord, AIMessage]: ...


class LLMAgent:
    """The agent `agent_id` of the `agents:` settings (with `variant`'s allowed_tools), generating with `backend`."""

    def __init__(
        self, agent_id: str, backend: ModelBackend, registry: ToolRegistry, variant: str | None = None
    ) -> None:
        root = get_settings(variant)
        config = next(a for a in root.agents.items if a.id == agent_id)
        self.config = config
        self.id = config.id
        self.backend = backend
        self.generation = root.generation_for(config)
        self.max_steps = root.max_steps_for(config)
        self.max_tool_calls = root.max_tool_calls_for(config)
        self.model_name = root.model_for(config).name
        # Raw-model log-probs are captured while generating only if this backend is also the scorer.
        self.logprobs = generation_logprob_mode(root.model_for(config))
        self.seed = root.project.seed
        # Permissions come from config only. The executor re-checks them on every call.
        self.allowed_tools = list(config.allowed_tools)
        self.effective_tools = registry.effective_tools(config.allowed_tools)
        self.tool_schemas = registry.schemas(self.effective_tools)
        if self.effective_tools:
            require(backend, "tool_calling")

    def initial_messages(self, problem: str, previous: list[str]) -> list[BaseMessage]:
        user = self.config.user_template.format(problem=problem, previous="\n\n".join(previous))
        return [SystemMessage(self.config.system_prompt), HumanMessage(user)]

    def step(self, messages: Sequence[BaseMessage], problem_id: str, step_id: int) -> tuple[StepRecord, AIMessage]:
        prompt_ids = self.backend.render(messages, self.tool_schemas or None)
        seed = derive_seed(self.seed, problem_id, self.id, step_id)
        request = GenerationRequest(prompt_ids, self.generation, seed=seed, n=1, logprobs=self.logprobs)
        (gen,) = self.backend.generate(request)
        parsed = self.backend.parse(gen.text)
        record = StepRecord(
            problem_id=problem_id,
            agent_id=self.id,
            step_id=step_id,
            input_messages=to_chat_dicts(messages),
            tools_offered=self.effective_tools,
            tool_schemas=self.tool_schemas,
            prompt_ids=prompt_ids,
            generation_params=to_plain(self.generation),
            seed=seed,
            token_ids=gen.token_ids,
            text=gen.text,
            token_logprobs=gen.token_logprobs,
            token_entropies=gen.token_entropies,
            scored_by=gen.scorer,
            generation_meta=gen.metadata() | {"compat": self.backend.info.compatibility_key()},
            content=parsed.content,
            reasoning=parsed.reasoning,
            tool_calls=parsed.tool_calls,
            invalid_tool_calls=parsed.invalid_tool_calls,
        )
        message = AIMessage(
            content=parsed.content,
            tool_calls=parsed.tool_calls,
            invalid_tool_calls=[{**c, "type": "invalid_tool_call"} for c in parsed.invalid_tool_calls],
            additional_kwargs={RAW_TEXT_KEY: gen.text},
        )
        logger.info(
            "agent step",
            extra={
                "problem_id": problem_id,
                "agent_id": self.id,
                "step_id": step_id,
                "generated_tokens": len(gen.token_ids),
                "tool_calls": [c["name"] for c in parsed.tool_calls],
                "invalid_tool_calls": len(parsed.invalid_tool_calls),
            },
        )
        return record, message

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.config.role,
            "model": self.model_name,
            "allowed_tools": self.allowed_tools,
            "effective_tools": self.effective_tools,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "generation": to_plain(self.generation),
        }
