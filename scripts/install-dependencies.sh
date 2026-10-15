#!/usr/bin/env bash
# Install dependencies with the torch extra that matches the local CUDA toolkit (nvcc).
#   scripts/install-dependencies.sh                 # torch only
#   scripts/install-dependencies.sh --vllm          # vLLM backend (Linux + NVIDIA): installs the `vllm`
#                                                   # extra, whose pinned torch replaces the cuXXX extra
#
# nvcc picks the extra (not nvidia-smi): nvidia-smi's "CUDA Version" is only the driver's upper bound.
set -euo pipefail
cd "$(dirname "$0")/.."
# Distro Pythons (/usr/lib/python3.x) can break torch's source introspection; use uv's own build.
export UV_PYTHON_PREFERENCE=only-managed

vllm=0
for arg in "$@"; do
    case "$arg" in
        --vllm) vllm=1 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

if (( vllm )); then
    [[ "$(uname -s)" == "Linux" ]] || { echo "--vllm needs Linux + an NVIDIA GPU" >&2; exit 1; }
    command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; vLLM needs an NVIDIA GPU" >&2
    extra=vllm
elif [[ "$(uname -s)" == "Darwin" ]]; then
    extra=mps
elif ! command -v nvcc >/dev/null; then
    extra=cpu
else
    cuda=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    case "$cuda" in
        13.*) extra=cu130 ;;
        12.[89]) extra=cu128 ;;
        12.[67]) extra=cu126 ;;
        *) echo "nvcc $cuda is unsupported: need CUDA >= 12.6" >&2; exit 1 ;;
    esac
    echo "nvcc reports CUDA $cuda"
fi

args=(--extra "$extra")

echo "uv sync ${args[*]}"
uv sync "${args[@]}"

[[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
