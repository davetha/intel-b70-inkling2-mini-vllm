#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:45' +%s)}
(( $(date +%s) + 400 < deadline ))
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
render_node=$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")
name="inkling2-coding-kernel-check-$$"
trap 'docker stop --time 5 "$name" >/dev/null 2>&1 || true' EXIT
source_mount=()
if (( $# )); then
    (( $# == 1 )) && test -f "$1" || { echo "Usage: $0 [code-sample.txt]" >&2; exit 2; }
    source_mount=(--mount "type=bind,source=$(realpath -- "$1"),target=/tmp/code-sample.txt,readonly")
fi
timeout --kill-after=10s 6m docker run --rm --name "$name" --network none \
    --cpus 4 --memory 18g --device "$render_node" \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    "${source_mount[@]}" --env PYTHONPATH=/lab:/lab/.work/python:/lab/.work/trellis-serve/core/src \
    --env PYTHONDONTWRITEBYTECODE=1 --env SYCL_UR_USE_LEVEL_ZERO_V2=1 --env SYCL_CACHE_PERSISTENT=0 \
    --env INKLING2_LIB=/lab/.work/build/inkling2-vllm.so --env EXL3_MOE_LIB=/lab/.work/build/inkling2-exl3-moe-mb32.so \
    --entrypoint /bin/bash "$runtime" -ec '
    /opt/venv/bin/python /lab/tests/check_grouped_qkv.py
    /opt/venv/bin/python /lab/tests/check_attention_prep.py
    INKLING2_CONTROL_LIB=/lab/.work/build/inkling2-loop-control.so /opt/venv/bin/python /lab/tests/check_draft_mix.py
    /opt/venv/bin/python /lab/tests/check_thinking_budget.py /lab/.work/build/inkling2-reasoning-budget.so
    if [[ -f /tmp/code-sample.txt ]]; then
        export INKLING2_LIB=/lab/.work/build/inkling2-vllm.so INKLING2_CONTROL_LIB=/lab/.work/build/inkling2-loop-control.so
        export INKLING2_PREP_LIB=/lab/.work/build/inkling2-fused-prep.so INKLING2_QKV_LIB=/lab/.work/build/inkling2-qkv.so
        export INKLING2_EXL3_SMALL_DPAS=1 INKLING2_RING_PREFIX=1 INKLING2_NATIVE_DEPTH_MIX=1 INKLING2_NATIVE_MERGE=1 INKLING2_TUNE_DENSE=1
        export INKLING2_GROUP_CONTROLLERS=1 INKLING2_FIXED_ROUTER=1 INKLING2_LOCAL_XMX=1 INKLING2_PAGED_XMX=1
        export INKLING2_SYCL_ROUTER=1 INKLING2_EXL3_PREFILL_MIN=256 INKLING2_EXL3_PAIR_MAX=8
        /opt/venv/bin/python /lab/tests/check_verifier.py /lab/models/exl3 /tmp/code-sample.txt
        /opt/venv/bin/python /lab/tests/check_ring_state.py /lab/models/exl3 /tmp/code-sample.txt
    fi
'
