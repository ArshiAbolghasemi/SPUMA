"""Backend-independent chat formatting and Qwen3 output parsing.

Both backends render prompts, decode tokens and parse tool calls through this module with the same
Hugging Face tokenizer, so a prompt/continuation pair has identical token ids whichever engine
generates or scores it.
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from spuma.common.utils import stable_hash
from spuma.models.backend import ParsedOutput

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
RAW_TEXT_KEY = "raw_text"  # AIMessage.additional_kwargs key holding the exact generated text


def to_chat_dicts(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """LangChain messages -> chat-template dicts. Assistant turns replay their exact generated text."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content})
        elif isinstance(m, AIMessage):
            raw = m.additional_kwargs.get(RAW_TEXT_KEY)
            if raw is not None:
                out.append({"role": "assistant", "content": raw})
            else:
                calls = [
                    {"type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": m.content, **({"tool_calls": calls} if calls else {})})
        else:
            raise TypeError(f"unsupported message type {type(m).__name__}")
    return out


def parse_qwen_output(text: str) -> ParsedOutput:
    """Split Qwen3 output into reasoning (<think>), Hermes-style <tool_call> JSON blocks and content."""
    reasoning = ""
    body = text
    if m := THINK_RE.match(body):
        reasoning, body = m.group(1).strip(), body[m.end() :]
    calls, invalid = [], []
    for i, block in enumerate(TOOL_CALL_RE.findall(body)):
        call_id = f"call_{i}"
        try:
            payload = json.loads(block.strip())
            name, args = payload["name"], payload.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(name, str) or not isinstance(args, dict):
                raise TypeError("name must be a string and arguments an object")
            calls.append({"name": name, "args": args, "id": call_id})
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            invalid.append({"name": None, "args": block.strip(), "id": call_id, "error": f"malformed tool call: {e}"})
    content = TOOL_CALL_RE.sub("", body).strip()
    return ParsedOutput(content=content, reasoning=reasoning, tool_calls=calls, invalid_tool_calls=invalid)


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

    def render(self, messages: Sequence[BaseMessage], tools: list[dict[str, Any]] | None) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            to_chat_dicts(messages),
            tools=tools or None,
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

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens]) + " …[truncated]"

    @staticmethod
    def parse(text: str) -> ParsedOutput:
        return parse_qwen_output(text)
