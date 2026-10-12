#!/usr/bin/env bash
# Compare inference backends on identical prompts; flags go to `python -m spuma.models.benchmark`.
#   scripts/run-benchmark.sh --backend transformers --backend vllm --problems 8
#   scripts/run-benchmark.sh --backend transformers                     # one backend
set -euo pipefail
cd "$(dirname "$0")/.."
# --no-sync: a plain `uv run` re-syncs without extras and would strip torch/vllm
exec uv run --no-sync --active python -m spuma.models.benchmark "$@"
