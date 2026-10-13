#!/usr/bin/env bash
# Run experiments and evaluation; all flags are forwarded to `python -m spuma.experiments`.
#   scripts/run-experiments.sh                                # every phase
#   scripts/run-experiments.sh --phase evaluate               # re-evaluate from saved artifacts
#   scripts/run-experiments.sh --limit 5                      # smoke run: 5 problems per split
set -euo pipefail
cd "$(dirname "$0")/.."
# --no-sync: a plain `uv run` re-syncs without extras and would strip torch
exec uv run --no-sync --active python -m spuma.experiments "$@"
