import pytest

from spuma.tools.worker import IsolatedRunner


@pytest.fixture(scope="session")
def runner():
    r = IsolatedRunner()
    yield r
    r.close()


def assert_langchain_tool(tool, name, schema, description_start):
    """Built with `langchain_core.tools.tool`: a StructuredTool with our name, docstring and pydantic schema."""
    from langchain_core.tools import StructuredTool

    assert isinstance(tool, StructuredTool)
    assert tool.name == name and tool.args_schema is schema
    assert tool.description.startswith(description_start)
