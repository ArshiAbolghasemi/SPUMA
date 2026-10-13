"""What a tool provider is: something that turns a `tools.registry` entry into a LangChain tool."""

from typing import Protocol

from langchain_core.tools import BaseTool


class ToolUnavailable(RuntimeError):
    """A provider cannot build a tool on this machine (e.g. no sandbox); the tool is disabled."""


class ToolProvider(Protocol):
    def build(self, name: str) -> BaseTool:
        """The LangChain tool (`langchain_core.tools.tool`) for `tools.registry.<name>`, or raise ToolUnavailable."""
        ...
