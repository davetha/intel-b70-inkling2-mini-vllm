#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
mode=${1:?ordinary or dspark}
context=${2:?context in tokens}
shift 2
port=${INKLING2_PORT:-8092}
batch=${INKLING2_BATCH:-512}
microbatch=${INKLING2_UBATCH:-128}
kv_type=${INKLING2_KV_TYPE:-f16}
draft_kv_type=${INKLING2_DRAFT_KV_TYPE:-f16}
image=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
model=/lab/models/unsloth/Inkling-2-Mini-UD-Q4_K_M.gguf
draft=/lab/models/unsloth/dspark-Inkling-2-Mini-Q8_0.gguf
binary=${INKLING2_SERVER:-/lab/.work/llama-unsloth/build/bin/llama-server}
name=inkling2-sycl-baseline

draft_args=()
case "$mode" in
    ordinary) ;;
    dspark)
        test -f "${draft#/lab/}"
        draft_args=(-md "$draft" -ngld 99 -ctkd "$draft_kv_type" -ctvd "$draft_kv_type"
                    --spec-type draft-dspark --spec-draft-n-max "${INKLING2_DRAFT_MAX:-3}")
        ;;
    *) echo 'Use ordinary or dspark.' >&2; exit 2 ;;
esac

test -f "${model#/lab/}"
test -x "${binary#/lab/}"
if systemctl --user is-active --quiet b70-qwen38-vllm.service; then
    echo 'The B70 is occupied by the production Qwen service.' >&2
    exit 1
fi

render_node=$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")
test -c "$render_node"
power_cap=(/sys/class/drm/"${render_node##*/}"/device/hwmon/hwmon*/power1_cap)
test "$(cat "${power_cap[0]}")" = 180000000

deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:44' +%s)}
seconds=$(( deadline - $(date +%s) ))
(( seconds > 60 ))
mkdir -p .work/sycl-program-cache

cid_dir=$(mktemp -d)
cleanup() {
    if [[ -s "$cid_dir/container" ]]; then
        docker stop --time 8 "$(<"$cid_dir/container")" >/dev/null 2>&1 || true
    fi
    if [[ -e "$cid_dir/container" ]]; then
        unlink "$cid_dir/container"
    fi
    rmdir "$cid_dir"
}
trap cleanup EXIT

timeout --kill-after=12s "${seconds}s" docker run --rm --init --name "$name" \
    --cidfile "$cid_dir/container" --network host --device "$render_node" --memory 20g --cpus 8 \
    --workdir /tmp --ulimit core=0 \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --mount "type=bind,source=$PWD/.work/sycl-program-cache,target=/sycl-cache" \
    --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu --env SYCL_CACHE_DIR=/sycl-cache \
    --env "SYCL_CACHE_PERSISTENT=${INKLING2_SYCL_CACHE:-0}" \
    --env "SYCL_UR_USE_LEVEL_ZERO_V2=${INKLING2_UR_LEVEL_ZERO_V2:-0}" \
    --env "GGML_SYCL_ENABLE_VMM=${INKLING2_SYCL_VMM:-0}" \
    --env "GGML_SYCL_ENABLE_GRAPH=${INKLING2_SYCL_GRAPH:-0}" \
    --env "LLAMA_DSPARK_DEV_FEATURES=${INKLING2_DSPARK_DEVICE:-0}" \
    --env "LLAMA_INKLING_BANDED_DECODE=${INKLING2_BANDED_DECODE:-0}" \
    --env "LLAMA_INKLING_SYCL_SWA_BIAS=${INKLING2_SWA_BIAS:-0}" \
    --env "GGML_SYCL_FA_MASK_SKIP=${INKLING2_MASK_SKIP:-0}" \
    --env "GGML_SYCL_INKLING_ESIMD=${INKLING2_ESIMD_PREFILL:-0}" \
    --env "GGML_SYCL_XMX_GATHER_TYPES=${INKLING2_XMX_TYPES:--1}" \
    --env "GGML_SCHED_DEBUG=${INKLING2_SCHED_DEBUG:-0}" \
    --env "GGML_SYCL_MMID_SCHED=${INKLING2_MMID_SCHED:-1}" \
    --env "LD_LIBRARY_PATH=$(dirname -- "$binary"):/opt/b70:/opt/b70/lib:/opt/venv/lib:/usr/local/lib" \
    --entrypoint "$binary" "$image" \
    --host 127.0.0.1 --port "$port" --device SYCL0 -ngl 99 --fit off \
    -m "$model" -c "$context" -np 1 -t 6 -b "$batch" -ub "$microbatch" \
    --flash-attn on -ctk "$kv_type" -ctv "$kv_type" --no-context-shift --cache-ram 0 \
    "${draft_args[@]}" "$@"
