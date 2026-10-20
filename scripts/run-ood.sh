#!/usr/bin/env bash
# OOD evaluation of a frozen SPN; all flags are forwarded to `python -m spuma.ood`.
#   scripts/run-ood.sh                                                          # every registered OOD evaluation
#   scripts/run-ood.sh --ood ood_morehopqa_math_ma_qwen_feedbackloop            # one by name (repeatable)
#   scripts/run-ood.sh --ood ood_morehopqa_math_ma_qwen_feedbackloop --limit 5  # smoke run
#   scripts/run-ood.sh --phase evaluate                                         # re-evaluate saved features
#   scripts/run-ood.sh --list                                                   # registered OOD evaluations
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --no-sync --active python -m spuma.ood "$@"
