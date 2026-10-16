"""Small dependency-free helpers shared across packages."""

import hashlib
import json
import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any


def stable_hash(data: Any, length: int = 12) -> str:
    """Short SHA-256 of the canonical JSON of `data` (used for config fingerprints)."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:length]


def derive_seed(base: int, *parts: object) -> int:
    """Deterministic per-call seed from the run seed and call coordinates (problem, agent, step, ...)."""
    key = ":".join(str(p) for p in (base, *parts)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def safe_name(identifier: str) -> str:
    """Filesystem-safe file name for an id such as `math/test/algebra/3`."""
    return identifier.replace("/", "__")


class MainThreadCalls:
    """Runs callables on the thread that created it (call `serve` from that thread's wait loop).

    Worker threads that need the main thread (math-verify's signal-based timeouts, non-thread-safe scorers)
    hand work over with `call` and block until it is done; on the owning thread `call` runs it directly.
    `close` fails anything still queued so no worker waits forever.
    """

    def __init__(self) -> None:
        self.owner = threading.get_ident()
        self.queue: queue.Queue[tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]] = queue.Queue()
        self.closed = False

    def call(self, fn: Callable[..., Any], *args: Any) -> Any:
        if threading.get_ident() == self.owner:
            return fn(*args)
        if self.closed:
            raise RuntimeError("main thread no longer serves calls")
        future: Future[Any] = Future()
        self.queue.put((future, fn, args))
        return future.result()

    def serve(self, timeout: float) -> None:
        """Run queued calls, waiting up to `timeout` seconds for the first one."""
        try:
            item = self.queue.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            future, fn, args = item
            try:
                future.set_result(fn(*args))
            except BaseException as e:  # noqa: BLE001 - handed back to the waiting worker
                future.set_exception(e)
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self.closed = True
        while True:
            try:
                future, _, _ = self.queue.get_nowait()
            except queue.Empty:
                return
            future.set_exception(RuntimeError("main thread stopped serving calls"))
