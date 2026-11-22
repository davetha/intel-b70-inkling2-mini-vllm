#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:44' +%s)}
seconds=$(( deadline - $(date +%s) ))
(( seconds > 60 ))
if (( seconds > 900 )); then
    seconds=900
fi
if systemctl --user is-active --quiet b70-qwen38-vllm.service || \
   [[ -n $(docker ps --filter name=inkling2 --format '{{.Names}}' | rg -v 'build' || true) ]]; then
    echo 'The B70 is occupied by another text experiment.' >&2
    exit 1
fi

binary=${INKLING2_BENCH:-/lab/.work/llama-unsloth/build/bin/llama-bench}
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
render_node=$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")
name="inkling2-prefill-bench-$$"
test -x "${binary#/lab/}"
trap 'docker stop --timeout 5 "$name" >/dev/null 2>&1 || true' EXIT
timeout --kill-after=10s "${seconds}s" docker run --rm --name "$name" --network none \
    --device "$render_node" --cpus 8 --memory 20g --workdir /tmp --ulimit core=0 \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu --env SYCL_CACHE_PERSISTENT=0 \
    --env SYCL_UR_USE_LEVEL_ZERO_V2=0 --env GGML_SYCL_ENABLE_VMM=0 \
    --env "GGML_SYCL_ENABLE_GRAPH=${INKLING2_SYCL_GRAPH:-0}" \
    --env "GGML_SYCL_MMID_SCHED=${INKLING2_MMID_SCHED:-1}" \
    --env "GGML_SYCL_XMX_GATHER_SHAPES=${INKLING2_XMX_SHAPES:-255}" \
    --env "GGML_SYCL_XMX_GATHER_TYPES=${INKLING2_XMX_TYPES:--1}" \
    --env "GGML_SYCL_DYNAMIC_PRECISION=${INKLING2_PRECISION:-f16}" \
    --env "LLAMA_INKLING_SYCL_SWA_BIAS=${INKLING2_SWA_BIAS:-0}" \
    --env "GGML_SYCL_FA_MASK_SKIP=${INKLING2_MASK_SKIP:-0}" \
    --env "GGML_SYCL_INKLING_ESIMD=${INKLING2_ESIMD_PREFILL:-0}" \
    --env "GGML_SCHED_DEBUG=${INKLING2_SCHED_DEBUG:-0}" \
    --env "LD_LIBRARY_PATH=$(dirname -- "$binary"):/opt/b70:/opt/b70/lib:/opt/venv/lib:/usr/local/lib" \
    --entrypoint "$binary" "$runtime" \
    -m /lab/models/unsloth/Inkling-2-Mini-UD-Q4_K_M.gguf -dev SYCL0 -ngl 99 -t 6 \
    -ctk q8_0 -ctv q8_0 -fa on -n 0 --no-warmup -o jsonl "$@"
