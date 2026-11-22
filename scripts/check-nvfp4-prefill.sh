#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
work=.work/llama-unsloth/nvfp4-prefill
image=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
builder=${INKLING2_BUILDER:-sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca}
render_node=$(readlink -f -- "${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}")
name="inkling2-nvfp4-check-$$"
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:44' +%s)}
(( $(date +%s) + 5 * 60 < deadline ))
if systemctl --user is-active --quiet b70-qwen38-vllm.service || \
   [[ -n $(docker ps --filter name=inkling2 --format '{{.Names}}' | rg -v 'build' || true) ]]; then
    echo 'The B70 is occupied by another text experiment.' >&2
    exit 1
fi

trap 'docker stop --timeout 5 "$name" >/dev/null 2>&1 || true' EXIT
timeout --kill-after=5s 30s docker run --rm --network none --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$PWD,target=/lab" --entrypoint c++ "$builder" \
    -O2 -std=c++17 -I "/lab/$work/source/ggml/include" /lab/tests/check_nvfp4_grouped.cpp \
    -L "/lab/$work/build/bin" -Wl,--allow-shlib-undefined -lggml -lggml-base -lggml-cpu \
    -o /lab/.work/build/check_nvfp4_grouped
timeout --kill-after=10s 4m docker run --rm --name "$name" --network none --device "$render_node" \
    --cpus 8 --memory 6g --workdir /tmp --ulimit core=0 \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --env ONEAPI_DEVICE_SELECTOR=level_zero:gpu --env SYCL_CACHE_PERSISTENT=0 \
    --env SYCL_UR_USE_LEVEL_ZERO_V2=0 --env GGML_SYCL_ENABLE_VMM=0 \
    --env "GGML_SYCL_XMX_GATHER_TYPES=${INKLING2_XMX_TYPES:--1}" \
    --env "GGML_SYCL_MMID_SCHED=${INKLING2_MMID_SCHED:-1}" \
    --env "LD_LIBRARY_PATH=/lab/$work/build/bin:/opt/b70:/opt/b70/lib:/opt/venv/lib:/usr/local/lib" \
    --entrypoint /lab/.work/build/check_nvfp4_grouped "$image"
