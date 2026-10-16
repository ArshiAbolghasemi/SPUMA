#!/usr/bin/env bash
# Install dependencies with the torch extra that matches the local CUDA toolkit (nvcc), else the driver (nvidia-smi).
#   scripts/install-dependencies.sh                 # torch only
#   scripts/install-dependencies.sh --vllm          # vLLM backend (Linux + NVIDIA): installs the `vllm`
#                                                   # extra, whose pinned torch replaces the cuXXX extra
#   scripts/install-dependencies.sh --index https://mirror/simple   # everything, torch too, from a PyPI mirror
#
# nvcc is preferred: nvidia-smi's "CUDA Version" is only the driver's upper bound. Without a toolkit that bound
# is enough, since torch wheels bundle their own CUDA runtime.
set -euo pipefail
cd "$(dirname "$0")/.."
# Distro Pythons (/usr/lib/python3.x) can break torch's source introspection; use uv's own build.
export UV_PYTHON_PREFERENCE=only-managed
# torch and vLLM wheels are GBs, and a caching mirror (Artifactory, Nexus) fetches an uncached file from
# upstream before it sends a byte; uv's 30 s default read timeout gives up first.
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1800}

vllm=0
index=""
while (( $# )); do
    case "$1" in
        --vllm) vllm=1 ;;
        --index) [[ $# -ge 2 ]] || { echo "--index needs a URL" >&2; exit 2; }; index=$2; shift ;;
        --index=*) index=${1#--index=} ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
    shift
done

if (( vllm )); then
    [[ "$(uname -s)" == "Linux" ]] || { echo "--vllm needs Linux + an NVIDIA GPU" >&2; exit 1; }
    command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; vLLM needs an NVIDIA GPU" >&2
    extra=vllm
elif [[ "$(uname -s)" == "Darwin" ]]; then
    extra=mps
else
    if command -v nvcc >/dev/null; then
        cuda=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        echo "nvcc reports CUDA $cuda"
    elif command -v nvidia-smi >/dev/null; then
        cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p')
        echo "no nvcc; nvidia-smi driver supports up to CUDA $cuda"
    else
        cuda=""
    fi
    case "$cuda" in
        "") extra=cpu ;;
        13.*) extra=cu130 ;;
        12.[89]) extra=cu128 ;;
        12.[67]) extra=cu126 ;;
        *) echo "CUDA $cuda is unsupported: need CUDA >= 12.6" >&2; exit 1 ;;
    esac
fi

args=(--extra "$extra")

if [[ -n "$index" ]]; then
    # `uv sync` would re-lock for every platform and CUDA extra, which needs download.pytorch.org.
    # Install only this machine's packages instead: every package at its uv.lock version, from the
    # mirror, with PyPI's own torch build. A dependency already pins the lock's torch, so the cuXXX/cpu/mps
    # paths prune it (and its CUDA wheels) from the export and install their own pin.
    torch=""
    case "$extra" in
        vllm) export_args=(--extra vllm) ;;  # its torch pin is PyPI's CUDA build already
        cu128)  # PyPI's Linux torch is CUDA 13 from 2.11; 2.10.0 is its last CUDA 12.8 build
            export_args=(--prune torch); torch=torch==2.10.0
            echo "note: the driver predates CUDA 13, so using PyPI torch 2.10.0 (CUDA 12.8), not the cu128 extra's 2.11.0" >&2 ;;
        cu126) echo "PyPI has no CUDA 12.6 torch build: update the driver to CUDA >= 12.8" >&2; exit 1 ;;
        *)
            export_args=(--prune torch); torch=$(sed -n "s/^$extra = \[\"\(torch==[^\"]*\)\"\]/\1/p" pyproject.toml)
            [[ "$extra" != cpu ]] || echo "note: PyPI's Linux torch is the CUDA build; it also runs on CPU" >&2 ;;
    esac
    pins=$(mktemp)
    trap 'rm -f "$pins"' EXIT
    uv export --frozen --no-hashes --no-emit-project --format requirements-txt "${export_args[@]}" -o "$pins" >/dev/null
    echo "using PyPI index $index; installing ${torch:-the $extra extra}"
    uv venv --allow-existing
    uv pip install --index-url "$index" -r "$pins" -e . $torch
    [[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
    exit
fi

echo "uv sync ${args[*]}"
uv sync "${args[@]}"

[[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
