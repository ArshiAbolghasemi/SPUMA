"""Backend-independent chat formatting and Qwen3 output parsing.

Both backends render prompts, decode tokens and parse outputs through this module with the same
Hugging Face tokenizer, so a prompt/continuation pair has identical token ids whichever engine
generates or scores it.
"""

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from spuma.common.utils import stable_hash
from spuma.models.backend import ParsedOutput

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)


def to_chat_dicts(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages (system + user) -> chat-template dicts."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        else:
            raise TypeError(f"unsupported message type {type(m).__name__}")
    return out


def parse_qwen_output(text: str) -> ParsedOutput:
    """Split Qwen3 output into reasoning (<think>) and content."""
    reasoning = ""
    body = text
    if m := THINK_RE.match(body):
        reasoning, body = m.group(1).strip(), body[m.end() :]
    return ParsedOutput(content=body.strip(), reasoning=reasoning)


class ChatFormatter:
    """Chat template rendering, (de)tokenization and parsing around one HF tokenizer."""

    def __init__(self, tokenizer: Any, enable_thinking: bool) -> None:
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking

    @property
    def template_hash(self) -> str:
        return stable_hash(getattr(self.tokenizer, "chat_template", None) or "")

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            to_chat_dicts(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self.encode(text)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    @staticmethod
    def parse(text: str) -> ParsedOutput:
        return parse_qwen_output(text)
