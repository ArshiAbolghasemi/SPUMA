"""Isolated execution boundary for math tools: a spawned single-process pool with a hard per-call timeout.

A call that exceeds its timeout kills the worker (SymPy cannot be interrupted cooperatively) and the
next call starts a fresh one. On Linux the worker also gets an address-space limit.
"""

import logging
import multiprocessing as mp
import sys
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class ToolTimeout(TimeoutError):
    pass


def _limit_memory(memory_mb: int) -> None:
    if sys.platform == "linux":
        import resource

        limit = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


class IsolatedRunner:
    def __init__(self, memory_mb: int = 2048) -> None:
        self.memory_mb = memory_mb
        self._pool: Any = None

    def _ensure(self) -> Any:
        if self._pool is None:
            ctx = mp.get_context("spawn")  # never fork a process holding torch/CUDA state
            self._pool = ctx.Pool(1, initializer=_limit_memory, initargs=(self.memory_mb,))
        return self._pool

    def run(self, fn: Callable[..., dict[str, Any]], timeout: float, **kwargs: Any) -> dict[str, Any]:
        result = self._ensure().apply_async(fn, kwds=kwargs)
        try:
            return result.get(timeout)
        except mp.TimeoutError:
            logger.warning("tool call timed out; restarting worker", extra={"fn": fn.__name__, "timeout": timeout})
            self.close()
            raise ToolTimeout(f"{fn.__name__} exceeded {timeout}s") from None
        except MemoryError:
            self.close()
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None
