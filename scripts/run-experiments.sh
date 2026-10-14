#!/usr/bin/env bash
# Run experiments and evaluation; all flags are forwarded to `python -m spuma.experiments`.
#   scripts/run-experiments.sh                                          # every registered experiment, every phase
#   scripts/run-experiments.sh --experiment math                        # only MATH (repeatable)
#   scripts/run-experiments.sh --experiment math --phase score --phase train --phase evaluate
#   scripts/run-experiments.sh --experiment math --limit 5              # smoke run: 5 problems per split
#   scripts/run-experiments.sh --list                                   # registered experiments
set -euo pipefail
cd "$(dirname "$0")/.."
# --no-sync: a plain `uv run` re-syncs without extras and would strip torch
exec uv run --no-sync --active python -m spuma.experiments "$@"
