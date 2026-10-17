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

# PyPI's vllm/torch are CUDA 13 builds; on a CUDA 12.9 driver swap in the same versions' +cu129 builds.
vllm_cuda_fix() {
    local p v f vllm_version pkgs=() idx=() py
    [[ "$vllm_cuda" == 12.9 ]] || return 0
    py=$(.venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
    for p in vllm torch torchvision torchaudio; do
        v=$(uv pip show "$p" | sed -n 's/^Version: \([^+]*\).*/\1/p')
        if [[ -n "$index" && "$p" != vllm ]]; then
            # The mirror's index links straight to download-r2.pytorch.org, so point at the wheel's mirror path instead.
            # ponytail: assumes an Artifactory mirror (.../api/pypi/<repo>/simple serving <repo>/whl/cu129)
            f=$(curl -fsSL "${index%/simple*}/whl/cu129/$p/" | grep -o "$p-$v\(%2B\|+\)cu129-$py-$py-manylinux[^\"#]*_$(uname -m)\.whl" | head -1)
            [[ -n "$f" ]] || { echo "no $p $v+cu129 wheel for $py on ${index%/simple*}/whl/cu129" >&2; exit 1; }
            pkgs+=("$p @ $(sed 's|/api/pypi/|/|' <<<"${index%/simple*}")/whl/cu129/$f")
        else
            pkgs+=("$p==$v+cu129")
        fi
        [[ "$p" != vllm ]] || vllm_version=$v
    done
    if [[ -n "$index" ]]; then idx=(--index-url "$index"); else idx=(--extra-index-url https://download.pytorch.org/whl/cu129); fi
    echo "driver supports CUDA $vllm_cuda (< 13): installing ${pkgs[*]}"
    uv pip install --reinstall --no-deps --index-strategy unsafe-best-match \
        "${idx[@]}" --extra-index-url "https://wheels.vllm.ai/$vllm_version/cu129" "${pkgs[@]}"
}
export UV_PYTHON_PREFERENCE=only-managed
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
    vllm_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p') || true
    case "$vllm_cuda" in
        ""|12.9|13.*|1[4-9].*) ;;
        *) echo "driver supports CUDA $vllm_cuda; vLLM needs a driver with CUDA >= 12.9" >&2; exit 1 ;;
    esac
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
    torch=""
    case "$extra" in
        vllm) export_args=(--extra vllm) ;;
        cu128)
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
    # A cold Artifactory cache can 404 a wheel it hasn't finished fetching upstream; uv doesn't retry 404s, so we do.
    for attempt in 1 2 3; do
        uv pip install --index-url "$index" -r "$pins" -e . $torch && break
        (( attempt < 3 )) || exit 1
        echo "install failed (attempt $attempt/3); retrying in 30s, the mirror may still be caching a wheel" >&2
        sleep 30
    done
    [[ "$extra" != vllm ]] || vllm_cuda_fix
    [[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
    exit
fi

echo "uv sync ${args[*]}"
uv sync "${args[@]}"
[[ "$extra" != vllm ]] || vllm_cuda_fix

[[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
