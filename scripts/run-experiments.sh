#!/usr/bin/env bash
# Run experiments and evaluation; all flags are forwarded to `python -m spuma.experiments`.
#   scripts/run-experiments.sh                                          # every registered experiment, every phase
#   scripts/run-experiments.sh --experiment math_ma_qwen_feedbackloop  # one experiment by name (repeatable)
#   scripts/run-experiments.sh --experiment math_ma_qwen_feedbackloop --phase score --phase train --phase evaluate
#   scripts/run-experiments.sh --experiment math_ma_qwen_feedbackloop --limit 5  # smoke run: 5 problems per split
#   scripts/run-experiments.sh --list                                   # registered experiments
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-sync --active python -m spuma.experiments "$@"
