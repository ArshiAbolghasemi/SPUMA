"""An agent = role prompt + model backend + generation settings. It answers in exactly one model call."""

import logging
from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from spuma.agents.trajectory import StepRecord
from spuma.common.config import to_plain
from spuma.common.utils import derive_seed
from spuma.config import get_settings
from spuma.models.backend import GenerationRequest, ModelBackend, generation_logprob_mode
from spuma.models.chat import to_chat_dicts

logger = logging.getLogger(__name__)


class Agent(Protocol):
    id: str

    def initial_messages(self, problem: str, previous: list[str]) -> list[BaseMessage]: ...

    def step(self, messages: Sequence[BaseMessage], problem_id: str, step_id: int) -> StepRecord: ...


class LLMAgent:
    """The agent `agent_id` of the `agents:` settings, generating with `backend`."""

    def __init__(self, agent_id: str, backend: ModelBackend) -> None:
        root = get_settings()
        config = next(a for a in root.agents.items if a.id == agent_id)
        self.config = config
        self.id = config.id
        self.backend = backend
        self.generation = root.generation_for(config)
        self.model_name = root.model_for(config).name
        # Raw-model log-probs are captured while generating only if this backend is also the scorer.
        self.logprobs = generation_logprob_mode(root.model_for(config))
        self.seed = root.project.seed

    def initial_messages(self, problem: str, previous: list[str]) -> list[BaseMessage]:
        user = self.config.user_template.format(problem=problem, previous="\n\n".join(previous))
        return [SystemMessage(self.config.system_prompt), HumanMessage(user)]

    def step(self, messages: Sequence[BaseMessage], problem_id: str, step_id: int) -> StepRecord:
        prompt_ids = self.backend.render(messages)
        seed = derive_seed(self.seed, problem_id, self.id, step_id)
        request = GenerationRequest(prompt_ids, self.generation, seed=seed, n=1, logprobs=self.logprobs)
        (gen,) = self.backend.generate(request)
        parsed = self.backend.parse(gen.text)
        record = StepRecord(
            problem_id=problem_id,
            agent_id=self.id,
            step_id=step_id,
            input_messages=to_chat_dicts(messages),
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
        )
        logger.info(
            "agent step",
            extra={
                "problem_id": problem_id,
                "agent_id": self.id,
                "step_id": step_id,
                "generated_tokens": len(gen.token_ids),
            },
        )
        return record

    def describe(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.config.role,
            "model": self.model_name,
            "generation": to_plain(self.generation),
        }
