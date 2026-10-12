"""What a tool provider is: something that turns a `tools.registry` entry into a LangChain tool."""

from typing import Protocol

from langchain_core.tools import BaseTool

from spuma.tools.config import ToolConfig


class ToolUnavailable(RuntimeError):
    """A provider cannot build a tool on this machine (e.g. no sandbox); the tool is disabled."""


class ToolProvider(Protocol):
    def build(self, name: str, config: ToolConfig) -> BaseTool:
        """Return a LangChain tool (built with `langchain_core.tools.tool`), or raise ToolUnavailable."""
        ...
