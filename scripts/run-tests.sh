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
    # Default: exclude tests that need a GPU or a model download (a later -m from the caller wins).
    args=(-m 'not gpu and not vllm and not nli' ${args[@]+"${args[@]}"})
fi
# --no-sync: a plain `uv run` re-syncs without extras and would strip torch.
# ${args[@]+...}: an empty array is "unbound" under `set -u` in macOS's bash 3.2.
exec uv run --no-sync --active python -m pytest ${args[@]+"${args[@]}"}
