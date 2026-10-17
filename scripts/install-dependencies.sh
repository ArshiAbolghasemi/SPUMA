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
#
# The lock's vllm/torch are CUDA 13 builds; on a CUDA 12.9 driver the same versions' cu129 builds replace them.
# Those installs use `uv pip`, so a later plain `uv sync` (or `uv run` without --no-sync) puts CUDA 13 back.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PYTHON_PREFERENCE=only-managed
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-1800}

die() { echo "$*" >&2; exit 1; }

vllm=0
index=""
while (( $# )); do
    case "$1" in
        --vllm) vllm=1 ;;
        --index) [[ $# -ge 2 ]] || die "--index needs a URL"; index=$2; shift ;;
        --index=*) index=${1#--index=} ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) die "unknown flag: $1" ;;
    esac
    shift
done

# "<package> <tag>" pairs: install the pytorch.org <tag> build of the locked version instead of the lock's build.
builds=()
cu129=0
if (( vllm )); then
    [[ "$(uname -s)" == "Linux" ]] || die "--vllm needs Linux + an NVIDIA GPU"
    command -v nvidia-smi >/dev/null || echo "warning: nvidia-smi not found; vLLM needs an NVIDIA GPU" >&2
    cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9]*\.[0-9]*\).*/\1/p') || true
    case "$cuda" in
        ""|13.*|1[4-9].*) ;;
        # torchcodec has no cu129 build of the locked version and its PyPI build needs CUDA 13 (libnvrtc.so.13);
        # vLLM only uses it to decode audio, so the CPU build is enough.
        12.9) cu129=1; builds=("torch cu129" "torchvision cu129" "torchaudio cu129" "torchcodec cpu") ;;
        *) die "driver supports CUDA $cuda; vLLM needs a driver with CUDA >= 12.9" ;;
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
        *) die "CUDA $cuda is unsupported: need CUDA >= 12.6" ;;
    esac
    # The lock takes torch from pytorch.org, which a mirror replaces.
    [[ -z "$index" ]] || builds=("torch $extra")
fi

if [[ -z "$index" && ${#builds[@]} -eq 0 ]]; then
    echo "uv sync --extra $extra"
    uv sync --extra "$extra"
    [[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
    exit
fi

locked() { uv export --frozen --no-hashes --no-emit-project --format requirements-txt --extra "$extra" "$@"; }
version() { sed -n "s/^$1==\([^+ ;]*\).*/\1/p" "$pins"; }

pins=$(mktemp)
trap 'rm -f "$pins"' EXIT
locked >"$pins"
uv venv --allow-existing
py=$(.venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
mirror=${index%/simple*}

specs=() sources=() drop=()
for b in "${builds[@]}"; do
    read -r p tag <<<"$b"
    v=$(version "$p")
    drop+=("$p")
    if [[ -z "$index" ]]; then
        specs+=("$p==$v+$tag")
        sources+=(--extra-index-url "https://download.pytorch.org/whl/$tag")
        continue
    fi
    # The mirror's pytorch index links straight to download-r2.pytorch.org, so install each wheel by its mirror URL.
    # ponytail: assumes Artifactory (.../api/pypi/<repo>/simple also serves <repo>/whl/<tag>, files at /<repo>/whl/<tag>)
    f=$(curl -fsSL "$mirror/whl/$tag/$p/" | grep -o "$p-$v\(%2B\|+\)$tag-$py-$py-manylinux[^\"#]*_$(uname -m)\.whl" | head -1)
    [[ -n "$f" ]] || die "no $p $v+$tag wheel for $py at $mirror/whl/$tag"
    specs+=("$p @ ${mirror/\/api\/pypi\//\/}/whl/$tag/$f")
done
if (( cu129 )); then
    v=$(version vllm)
    specs+=("vllm==$v+cu129")
    sources+=(--extra-index-url "https://wheels.vllm.ai/$v/cu129")
    # vllm also pulls CUDA 13 pins (cuda-python, nvidia-cuda-runtime, ...); let the resolver pick cu12 ones.
    drop+=(vllm triton 'cuda-[a-z-]+' 'nvidia-[a-z0-9-]+')
fi
# Prune torch so its CUDA libs come from the builds above, not the lock's.
if (( ${#drop[@]} )); then
    locked --prune torch | grep -vE "^($(IFS='|'; echo "${drop[*]}"))==" >"$pins"
fi

idx=()
[[ -z "$index" ]] || idx=(--index-url "$index")
echo "installing the $extra extra${index:+ from $index}${specs[*]:+ with ${specs[*]}}"
uv pip install "${idx[@]}" --index-strategy unsafe-best-match "${sources[@]}" -r "$pins" -e . "${specs[@]}"
[[ -f .env ]] || echo "note: cp .env.example .env and set HF_TOKEN"
