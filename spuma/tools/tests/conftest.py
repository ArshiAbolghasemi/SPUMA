import pytest

from spuma.tools.registry import build_tool_registry, default_providers
from spuma.tools.worker import IsolatedRunner


@pytest.fixture(scope="session")
def runner():
    r = IsolatedRunner()
    yield r
    r.close()


@pytest.fixture
def registry(config, runner):
    return build_tool_registry(config.tools, default_providers(runner))
