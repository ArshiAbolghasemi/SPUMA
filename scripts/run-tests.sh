#!/usr/bin/env bash
# Run the test suite; all flags are forwarded to pytest.
#   scripts/run-tests.sh                                  # offline suite (no GPU, no model download, no vLLM, no NLI model)
#   scripts/run-tests.sh --cov                            # + coverage report
#   scripts/run-tests.sh --all                            # include gpu (Qwen2.5-7B-Instruct), vllm and nli tests
#   scripts/run-tests.sh -m vllm                          # only the vLLM / cross-backend tests (Linux + GPU)
#   scripts/run-tests.sh -m nli                           # only the real DeBERTa-MNLI semantic-entropy test
#   scripts/run-tests.sh spuma/spn -k histogram -x
set -euo pipefail
cd "$(dirname "$0")/.."
all=0
args=()
for arg in "$@"; do
    case "$arg" in
        --all) all=1 ;;
        --cov) args+=(--cov=spuma --cov-report=term-missing) ;;
        *) args+=("$arg") ;;
    esac
done
if (( all )); then
    export SPUMA_TESTS_ALL=1
else
    args=(-m 'not gpu and not vllm and not nli' ${args[@]+"${args[@]}"})
fi
exec uv run --no-sync --active python -m pytest ${args[@]+"${args[@]}"}
