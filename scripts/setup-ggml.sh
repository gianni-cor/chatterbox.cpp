#!/usr/bin/env bash
# Clone ggml into ./ggml, check out the commit this repo is pinned against,
# and apply the Chatterbox Metal + (optional) OpenCL patches.  Re-running
# resets the ggml worktree to the pin and reapplies both — local edits under
# ./ggml are discarded.
#
# Update GGML_COMMIT when patches are re-generated against a newer upstream
# ggml; this file is the single source of truth for the pin.

set -euo pipefail

# -----------------------------------------------------------------------------
# The upstream ggml commit that patches/ggml-metal-chatterbox-ops.patch and
# patches/ggml-opencl-chatterbox-ops.patch are authored against.
# -----------------------------------------------------------------------------
GGML_COMMIT="58c38058"
GGML_URL="https://github.com/ggml-org/ggml.git"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "chatterbox.cpp: setting up ggml at pinned commit ${GGML_COMMIT}"

if [ ! -d ggml/.git ]; then
    echo "  → cloning ${GGML_URL}"
    git clone "$GGML_URL" ggml
fi

cd ggml

echo "  → resetting to ${GGML_COMMIT} (discarding uncommitted changes under ./ggml)"
git fetch origin 2>/dev/null || true
git reset --hard "$GGML_COMMIT"
# Remove untracked files (e.g. left over from a previously applied patch) so
# reapply is deterministic; ggml/ is not intended for long-lived local work.
git clean -fdq

echo "  → applying patches/ggml-metal-chatterbox-ops.patch"
git apply "$REPO_ROOT/patches/ggml-metal-chatterbox-ops.patch"

echo "  → applying patches/ggml-opencl-chatterbox-ops.patch"
git apply "$REPO_ROOT/patches/ggml-opencl-chatterbox-ops.patch"

N_METAL="$(git status --porcelain src/ggml-metal/ 2>/dev/null | wc -l | tr -d ' ')"
N_OPENCL="$(git status --porcelain include/ggml-opencl.h src/ggml-opencl/ 2>/dev/null | wc -l | tr -d ' ')"
echo "  → ok (Metal: ${N_METAL} paths touched, OpenCL: ${N_OPENCL} paths touched under ggml/)"
echo
echo "ggml is ready.  Next:"
echo "  Metal:   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON"
echo "  OpenCL:  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENCL=ON"
echo "  cmake --build build -j\$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
