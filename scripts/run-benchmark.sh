#!/usr/bin/env bash
# Compare inference backends on identical prompts; flags go to `python -m spuma.models.benchmark`.
#   scripts/run-benchmark.sh --backend transformers --backend vllm --problems 8
#   scripts/run-benchmark.sh --backend transformers                     # one backend
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-sync --active python -m spuma.models.benchmark "$@"
