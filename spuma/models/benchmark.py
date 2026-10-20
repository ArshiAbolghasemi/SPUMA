"""Backend performance comparison on identical prompts and settings.

    uv run python -m spuma.models.benchmark --backend transformers --backend vllm --problems 8

For each backend (loaded one at a time, released before the next) it measures, on the same MATH
test problems rendered with the Solver's prompt:
  - main generation with generation-time scores (`scoring_backend: same` path): requests/s, tokens/s, latency;
  - teacher-forced PE/NLL scoring of those continuations;
  - semantic-entropy sampling (n samples per prompt, one batched call);
  - peak GPU memory (when CUDA is available).
Results are logged and written to artifacts/<experiment>/benchmark-<run>.json.
Latency and memory differ across engines by design; outputs of different engines are not compared here
(see spuma/models/tests/test_cross_backend.py for numerical agreement on fixed sequences).
"""

import argparse
import dataclasses
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

import torch
from langchain_core.messages import HumanMessage, SystemMessage

from spuma.common.logs import setup_logging
from spuma.config import PROJECT_ROOT, SETTINGS_FILE_ENV, get_settings
from spuma.dataset.registry import get_dataset
from spuma.experiments.artifacts import FileArtifactStore
from spuma.models.backend import GenerationRequest, ModelBackend
from spuma.models.engines import get_backend, release_backends

logger = logging.getLogger("spuma.models.benchmark")


def _gpu() -> dict[str, float]:

    if not torch.cuda.is_available():
        return {}
    return {"gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}


def benchmark_backend(backend: ModelBackend, prompts: list[list[int]]) -> dict[str, Any]:
    config = get_settings()
    gen_cfg = config.model.generation
    se = config.uncertainty.semantic_entropy
    seed = config.project.seed

    start = time.perf_counter()
    mains = [r[0] for r in backend.generate_batch([GenerationRequest(p, gen_cfg, seed, 1, "full") for p in prompts])]
    gen_s = time.perf_counter() - start
    tokens = sum(m.num_tokens for m in mains)

    start = time.perf_counter()
    for p, m in zip(prompts, mains, strict=True):
        backend.score(p, m.token_ids, full_vocabulary=True)
    score_s = time.perf_counter() - start

    se_params = dataclasses.replace(gen_cfg, temperature=se.temperature, top_p=se.top_p, top_k=None)
    start = time.perf_counter()
    samples = backend.generate_batch([GenerationRequest(p, se_params, seed, se.n) for p in prompts])
    se_s = time.perf_counter() - start
    return {
        "backend": backend.name,
        "info": backend.info.compatibility_key() | {"library_version": backend.info.library_version},
        "requests": len(prompts),
        "generation": {
            "seconds": gen_s,
            "requests_per_s": len(prompts) / gen_s,
            "tokens_per_s": tokens / gen_s,
            "mean_latency_s": gen_s / len(prompts),
            "generated_tokens": tokens,
        },
        "teacher_forced_scoring": {"seconds": score_s, "per_request_s": score_s / len(prompts)},
        "semantic_sampling": {
            "seconds": se_s,
            "n": se.n,
            "tokens_per_s": sum(r.num_tokens for rs in samples for r in rs) / se_s,
        },
        **_gpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", action="append", choices=["transformers", "vllm"], required=True)
    parser.add_argument("--problems", type=int, default=8)
    parser.add_argument(
        "--config", help="path to the config file (default: config/math_ma_qwen2_5_7b_feedbackloop.yaml)"
    )
    args = parser.parse_args()

    if args.config:
        os.environ[SETTINGS_FILE_ENV] = args.config
    config = get_settings()
    setup_logging("benchmark")
    problems = get_dataset().load()["test"][: args.problems]
    agent = config.agents.enabled[0]
    results = []
    for kind in args.backend:
        backend = get_backend(config.model, kind)
        prompts = [
            backend.render(
                [
                    SystemMessage(agent.system_prompt),
                    HumanMessage(agent.user_template.format(problem=p.problem, previous="")),
                ]
            )
            for p in problems
        ]
        result = benchmark_backend(backend, prompts)
        logger.info("benchmark", extra=result)
        results.append(result)
        release_backends()
    store = FileArtifactStore(PROJECT_ROOT / config.experiment.output_dir / config.experiment.name)
    path = store.write_json(f"benchmark-{datetime.now(UTC):%Y%m%dT%H%M%S}.json", results)
    logger.info("benchmark written", extra={"path": str(path)})


if __name__ == "__main__":
    main()
