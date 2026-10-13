import threading
import time

import pytest

from spuma.models.backend import GenerationRequest, GenerationResult
from spuma.models.batching import MicroBatcher
from spuma.models.config import GenerationConfig


class _Engine:
    """Records how requests arrive; answers each with its own seed so routing can be checked."""

    name, info, capabilities, scorer_id = "fake", None, None, "fake:0"

    def __init__(self, fail=False):
        self.batch_sizes, self.fail, self.threads = [], fail, set()

    def generate_batch(self, requests):
        self.threads.add(threading.current_thread().name)
        self.batch_sizes.append(len(requests))
        time.sleep(0.02)  # engine time, during which more requests queue up
        if self.fail:
            raise RuntimeError("engine failed")
        return [
            [GenerationResult([r.seed], str(r.seed), "stop", 1, "fake", "m", "t", r.seed, {}) for _ in range(r.n)]
            for r in requests
        ]


def _request(seed, n=1):
    return GenerationRequest([1, 2, 3], GenerationConfig(temperature=0.7), seed=seed, n=n)


def test_concurrent_calls_are_batched_and_routed():
    engine = _Engine()
    batcher = MicroBatcher(engine, window_s=0.05)
    results, start = {}, threading.Barrier(8)

    def call(seed):
        start.wait()
        results[seed] = batcher.generate(_request(seed))

    threads = [threading.Thread(target=call, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    batcher.stop()
    assert {s: r[0].seed for s, r in results.items()} == {s: s for s in range(8)}  # each caller gets its own
    assert sum(engine.batch_sizes) == 8 and len(engine.batch_sizes) < 8  # fewer engine calls than requests
    assert engine.threads == {"spuma-batcher-fake"}  # the engine is only ever called from one thread


def test_generate_batch_keeps_n_samples_per_request():
    batcher = MicroBatcher(_Engine())
    out = batcher.generate_batch([_request(1, n=10), _request(2, n=3)])
    batcher.stop()
    assert [len(r) for r in out] == [10, 3]


def test_engine_errors_reach_every_caller():
    batcher = MicroBatcher(_Engine(fail=True))
    with pytest.raises(RuntimeError, match="engine failed"):
        batcher.generate(_request(0))
    batcher.stop()


def test_delegates_text_methods(scripted_backend_cls):
    backend = scripted_backend_cls([])
    batcher = MicroBatcher(backend)
    assert batcher.count_tokens("abc") == 3 and batcher.scorer_id == backend.scorer_id
    batcher.close()
    assert backend.closed
