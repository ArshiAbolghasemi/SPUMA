"""Test-run policy for every test (registered in pyproject: `-p spuma.common.tests.fixtures`)."""

import os
import re

import pytest

HEAVY_MARKERS = ("gpu", "vllm", "nli")


def pytest_collection_modifyitems(config, items):
    """Model-downloading tests (gpu, vllm, nli) run only when explicitly selected: their marker is named
    positively in `-m` (e.g. `-m nli`, `-m "gpu or vllm"`), or SPUMA_TESTS_ALL=1 (run-tests.sh --all).

    Any other invocation, including a bare `pytest` or a `-m` filter that simply does not mention a
    marker, skips them, so no test run downloads a model by accident.
    """

    if os.environ.get("SPUMA_TESTS_ALL") == "1":
        return
    expr = config.option.markexpr or ""
    selected = {m for m in HEAVY_MARKERS if re.search(rf"(?<!not )\b{m}\b", expr)}
    for item in items:
        heavy = [m for m in HEAVY_MARKERS if m in item.keywords]
        if heavy and not selected.intersection(heavy):
            reason = f"downloads a model ({', '.join(heavy)}): select with -m {heavy[0]} or run-tests.sh --all"
            item.add_marker(pytest.mark.skip(reason=reason))
