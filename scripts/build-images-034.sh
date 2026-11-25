#!/usr/bin/env bash
# Build the vLLM 0.34.0 runtime and oneAPI builder images used by the b70-prod scripts.
# Recipe: Danmoreng's pinned Qwen direct-XPU build plus patches/qwen-build-vllm-034-inkling-runtime.patch
# (official vllm/vllm-openai-xpu v0.34.0 base, no vLLM source overrides). Takes 1-2 hours.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/b70-prod-env.sh
commit=0356a370da5f73f6b0a4eab613d7103a2356b513
src=.work/qwen-build-034
if [[ ! -d $src/.git ]]; then
    git clone https://github.com/Danmoreng/intel-b70-qwen38-vllm.git "$src"
fi
git -C "$src" checkout --quiet --detach "$commit"
if ! git -C "$src" apply --reverse --check "$PWD/patches/qwen-build-vllm-034-inkling-runtime.patch" 2>/dev/null; then
    git -C "$src" apply "$PWD/patches/qwen-build-vllm-034-inkling-runtime.patch"
fi
cd "$src"
# build-engine.py keeps Danmoreng's Europe/Berlin daytime build window (ends 22:45).
python3 scripts/build-engine.py --tag "$INKLING2_RUNTIME" --output benchmark-results/inkling-034-runtime \
    --engine-dir engine/direct-xpu-034-inkling --dockerfile docker/direct-xpu-034-inkling/Dockerfile
docker build --build-arg "BASE_IMAGE=$INKLING2_RUNTIME" --tag "$INKLING2_BUILDER" \
    -f benchmarks/experiments/m17-oneapi-recompile/Dockerfile.compiler benchmarks/experiments/m17-oneapi-recompile
