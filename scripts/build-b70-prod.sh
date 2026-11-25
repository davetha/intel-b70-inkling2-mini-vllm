#!/usr/bin/env bash
# Build every kernel library the b70-prod profile loads into .work/build/.
# The first step parity-checks inkling2-vllm.so on the GPU, so give it the B70.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/b70-prod-env.sh
docker image inspect "$INKLING2_BUILDER" "$INKLING2_RUNTIME" >/dev/null

bash scripts/check-sycl.sh                    # fetches Trellis Serve, builds + checks inkling2-vllm.so
# The launcher requires the EXL3 library even with IQR experts; its parity test needs models/exl3, so skip it.
INKLING2_EXL3_TILE_ROWS=32 INKLING2_EXL3_TEST=0 bash scripts/check-sycl.sh exl3
bash scripts/build-loop-control.sh            # loop-control, qkv, fused-prep, reasoning-budget
bash scripts/build-iqr-moe.sh                 # IQR 4-bit experts
bash scripts/build-rel-attention.sh           # prefill attention with in-kernel relative bias
bash scripts/build-decode-xmx.sh              # 256-GRF decode XMX attention
ls -la .work/build/*.so
