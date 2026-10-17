#!/usr/bin/env bash
# Install dependencies with the torch extra that matches the local CUDA toolkit (nvcc), else the driver (nvidia-smi).
#   scripts/install-dependencies.sh                 # torch only
#   scripts/install-dependencies.sh --vllm          # vLLM backend (Linux + NVIDIA): installs the `vllm`
#                                                   # extra, whose pinned torch replaces the cuXXX extra
#   scripts/install-dependencies.sh --index https://mirror/simple   # everything from a PyPI mirror; torch from its
#                                                   # pytorch.org proxy (Artifactory layout)
#
# nvcc is preferred: nvidia-smi's "CUDA Version" is only the driver's upper bound. Without a toolkit that bound
# is enough, since torch wheels bundle their own CUDA runtime.
set -euo pipefail
cd "$(dirname "$0")/.."

# PyPI's vllm/torch are CUDA 13 builds; on a CUDA 12.9 driver swap in the same versions' +cu129 builds.
vllm_cuda_fix() {
    local p v vllm_version pkgs=()
    [[ "$vllm_cuda" == 12.9 ]] || return 0
    for p in vllm torch torchvision torchaudio; do
        v=$(uv pip show "$p" | sed -n 's/^Version: \([^+]*\).*/\1/p')
        pkgs+=("$p==$v+cu129")
        [[ "$p" != vllm ]] || vllm_version=$v
    done
    echo "driver supports CUDA $vllm_cuda (< 13): installing ${pkgs[*]}"
    uv pip install --reinstall --no-deps --index-strategy unsafe-best-match \
        --extra-index-url "https://wheels.vllm.ai/$vllm_version/cu129" --extra-index-url https://download.pytorch.org/whl/cu129 \
        "${pkgs[@]}"
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
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
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
    # Which pytorch.org build to take instead of PyPI's torch ("" = PyPI's is right).
    case "$extra" in
        vllm) tag=""; [[ "${vllm_cuda:-}" != 12.9 ]] || tag=cu129 ;;
        mps) tag="" ;;
        *) tag=$extra ;;
    esac
    pins=$(mktemp)
    trap 'rm -f "$pins"' EXIT
    uv export --frozen --no-hashes --no-emit-project --format requirements-txt "${args[@]}" -o "$pins" >/dev/null
    uv venv --allow-existing
    torch=() vllm_index=()
    if [[ -n "$tag" ]]; then
        # The mirror's pytorch index links straight to download-r2.pytorch.org, so install each wheel by its mirror URL.
        # ponytail: assumes Artifactory (.../api/pypi/<repo>/simple also serves <repo>/whl/<tag>, files at /<repo>/whl/<tag>)
        mirror=${index%/simple*}
        py=$(.venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
        pkgs=(torch); [[ "$extra" != vllm ]] || pkgs+=(torchvision torchaudio)
        for p in "${pkgs[@]}"; do
            v=$(sed -n "s/^$p==\([^+ ;]*\).*/\1/p" "$pins")
            f=$(curl -fsSL "$mirror/whl/$tag/$p/" | grep -o "$p-$v\(%2B\|+\)$tag-$py-$py-manylinux[^\"#]*_$(uname -m)\.whl" | head -1)
            [[ -n "$f" ]] || { echo "no $p $v+$tag wheel for $py at $mirror/whl/$tag" >&2; exit 1; }
            torch+=("$p @ ${mirror/\/api\/pypi\//\/}/whl/$tag/$f")
        done
        if [[ "$tag" == cu129 ]]; then
            v=$(sed -n 's/^vllm==\([^ ;]*\).*/\1/p' "$pins")
            torch+=("vllm==$v+cu129")
            vllm_index=(--index-strategy unsafe-best-match --extra-index-url "https://wheels.vllm.ai/$v/cu129")
        fi
        # Prune torch so its CUDA libs come from the build above, not the locked PyPI build's.
        uv export --frozen --no-hashes --no-emit-project --format requirements-txt "${args[@]}" --prune torch \
            | grep -vE '^(torchvision|torchaudio|vllm)==' >"$pins"
    fi
    echo "using PyPI index $index; installing the $extra extra${tag:+ (torch: pytorch.org $tag build)}"
    uv pip install --index-url "$index" "${vllm_index[@]}" -r "$pins" -e . "${torch[@]}"
    [[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
    exit
fi

echo "uv sync ${args[*]}"
uv sync "${args[@]}"
[[ "$extra" != vllm ]] || vllm_cuda_fix

[[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
