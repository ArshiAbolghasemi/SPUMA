"""Dynamic micro-batching: one engine shared by several concurrently running problem workflows.

Workflows call `generate` / `generate_batch` from their own threads as usual. The batcher collects the
requests that arrive within a short window and sends them to the wrapped backend as ONE
`generate_batch` call from a single dispatcher thread, so vLLM decodes many problems' sequences
together (its offline `LLM` must not be called from several threads at once). Results go back to
each caller unchanged; requests keep their own seeds and sampling parameters.

Tokenizer calls (`render`, `count_tokens`) are serialized: Hugging Face fast tokenizers
can fail with "Already borrowed" when used from several threads at once.
"""

import logging
import queue
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any

from langchain_core.messages import BaseMessage

from spuma.models.backend import GenerationRequest, GenerationResult, ModelBackend, ParsedOutput, TokenScores

logger = logging.getLogger(__name__)

BATCH_WINDOW_S = 0.01  # how long the dispatcher waits for more requests after the first one arrives
_STOP = object()


class MicroBatcher:
    """A `ModelBackend` that batches concurrent generation requests onto `backend`."""

    def __init__(self, backend: ModelBackend, window_s: float = BATCH_WINDOW_S) -> None:
        self.backend = backend
        self.name, self.info = backend.name, backend.info
        self.capabilities, self.scorer_id = backend.capabilities, backend.scorer_id
        self.window_s = window_s
        self.batches = 0  # dispatched engine calls (for logs and tests)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._tokenizer_lock = threading.Lock()
        self._thread = threading.Thread(target=self._dispatch, name=f"spuma-batcher-{backend.name}", daemon=True)
        self._thread.start()

    # --- generation: batched -------------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> list[GenerationResult]:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[list[GenerationResult]]:
        futures: list[Future[list[GenerationResult]]] = []
        for request in requests:
            future: Future[list[GenerationResult]] = Future()
            self._queue.put((request, future))
            futures.append(future)
        return [f.result() for f in futures]

    def _dispatch(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            batch, stopping = [first], False
            deadline = time.monotonic() + self.window_s
            while (left := deadline - time.monotonic()) > 0:
                try:
                    item = self._queue.get(timeout=left)
                except queue.Empty:
                    break
                if item is _STOP:
                    stopping = True
                    break
                batch.append(item)
            self._run(batch)
            if stopping:
                return

    def _run(self, batch: list[tuple[GenerationRequest, Future[list[GenerationResult]]]]) -> None:
        start = time.perf_counter()
        try:
            results = self.backend.generate_batch([request for request, _ in batch])
        except BaseException as e:  # noqa: BLE001 - every waiting caller gets the engine's error
            for _, future in batch:
                future.set_exception(e)
            return
        for (_, future), result in zip(batch, results, strict=True):
            future.set_result(result)
        self.batches += 1
        logger.debug(
            "batched generation",
            extra={
                "requests": len(batch),
                "sequences": sum(r.n for r, _ in batch),
                "seconds": round(time.perf_counter() - start, 3),
            },
        )

    def stop(self) -> None:
        """Stop the dispatcher (after the requests already queued); the wrapped engine stays loaded."""
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join()

    def close(self) -> None:
        self.stop()
        self.backend.close()

    # --- everything else: delegated ------------------------------------------------------------

    def render(self, messages: Sequence[BaseMessage]) -> list[int]:
        with self._tokenizer_lock:
            return self.backend.render(messages)

    def count_tokens(self, text: str) -> int:
        with self._tokenizer_lock:
            return self.backend.count_tokens(text)

    def score(self, prompt_ids: list[int], token_ids: list[int], full_vocabulary: bool) -> TokenScores:
        return self.backend.score(prompt_ids, token_ids, full_vocabulary)

    def parse(self, text: str) -> ParsedOutput:
        return self.backend.parse(text)
