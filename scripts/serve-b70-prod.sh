#!/usr/bin/env bash
# The b70-prod serving profile: IQR GPTQ-4 experts, DSpark + n-gram copy drafts, fused prefill and decode kernels.
# Every setting below can be overridden from the environment; extra arguments go to vLLM.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/b70-prod-env.sh
export INKLING2_MODEL_DIR=${INKLING2_MODEL_DIR:-models/iqr4-unc} INKLING2_EXPERTS=${INKLING2_EXPERTS:-iqr4}
export INKLING2_HOST=${INKLING2_HOST:-127.0.0.1} INKLING2_VLLM_PORT=${INKLING2_VLLM_PORT:-8093}
export INKLING2_ONEDNN_DETERMINISTIC=${INKLING2_ONEDNN_DETERMINISTIC:-1}
export INKLING2_NGRAM_DRAFT=${INKLING2_NGRAM_DRAFT:-1} INKLING2_NGRAM_NOCOPY=${INKLING2_NGRAM_NOCOPY:-1}
export INKLING2_NGRAM_MIN=${INKLING2_NGRAM_MIN:-3} INKLING2_NGRAM_MAX=${INKLING2_NGRAM_MAX:-32}
export INKLING2_NGRAM_DEPTH=${INKLING2_NGRAM_DEPTH:-7} INKLING2_DRAFT_MAX=${INKLING2_DRAFT_MAX:-3}
export INKLING2_PREFILL_PROFILE=${INKLING2_PREFILL_PROFILE:-optimized} INKLING2_VLLM_BATCH=${INKLING2_VLLM_BATCH:-4096}
export INKLING2_LOCAL_BIAS_TILE=${INKLING2_LOCAL_BIAS_TILE:-64,256,8} INKLING2_PREFILL_GLUE=${INKLING2_PREFILL_GLUE:-1}
export INKLING2_PREFILL_REL_BIAS=${INKLING2_PREFILL_REL_BIAS:-1} INKLING2_PREFILL_GLOBAL_TABLE=${INKLING2_PREFILL_GLOBAL_TABLE:-1}
export INKLING2_DECODE_XMX256=${INKLING2_DECODE_XMX256:-1} INKLING2_DECODE_CASTS=${INKLING2_DECODE_CASTS:-1}
export INKLING2_DECODE_XMX_LIB=${INKLING2_DECODE_XMX_LIB:-/lab/.work/build/inkling2-decode-xmx.so}
for lib in inkling2-iqr-moe inkling2-rel-attention inkling2-decode-xmx; do
    test -s ".work/build/$lib.so" || { echo "Missing .work/build/$lib.so: run scripts/build-b70-prod.sh" >&2; exit 1; }
done
test -s "$INKLING2_MODEL_DIR/iqr_config.json" || { echo "Missing IQR checkpoint in $INKLING2_MODEL_DIR" >&2; exit 1; }
exec bash scripts/serve-coding-agent.sh dspark "$@"
