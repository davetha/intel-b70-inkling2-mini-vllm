#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
source_commit=8f8e2fccf0c82c9715b774812f30d9d52f78a5fb
builder=${INKLING2_BUILDER:-sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca}
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
render_node=${INKLING2_RENDER_NODE:-/dev/dri/by-path/pci-0000:03:00.0-render}
vendor="$PWD/.work/trellis-serve"
output="$PWD/.work/build"
patch_file="$PWD/patches/trellis-vllm-block-table.patch"
source_file=xpu/csrc/inkling2_ops.sycl
mode=${1:-paged}
minutes=9
case "$mode" in
    paged) ;;
    exl3)
        patch_file="$PWD/patches/trellis-inkling-exl3-mcg-situ.patch"
        source_file=xpu/csrc/exl3_moe.sycl
        minutes=12
        ;;
    *) echo 'Use paged or exl3.' >&2; exit 2 ;;
esac
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:45' +%s)}
if (( $(date +%s) + minutes * 60 >= deadline )); then
    echo "Der SYCL-Test braucht bis zu $minutes Minuten vor 22:45 Europe/Berlin." >&2
    exit 1
fi

docker image inspect "$builder" "$runtime" >/dev/null
test -c "$render_node"
mkdir -p -- "$output"
exec 9>"$output/check-sycl.lock"
flock -n 9

if [[ ! -d "$vendor/.git" ]]; then
    git init -q "$vendor"
fi
if ! git -C "$vendor" cat-file -e "$source_commit^{commit}" 2>/dev/null; then
    timeout 30s git -C "$vendor" fetch --no-tags --depth 1 \
        https://github.com/sybil-solutions/trellis-serve.git "$source_commit"
fi
git -C "$vendor" checkout --quiet --detach "$source_commit"
if git -C "$vendor" diff --quiet -- "$source_file"; then
    git -C "$vendor" apply "$patch_file"
fi
git -C "$vendor" diff --exit-code --cached
git -C "$vendor" diff -- "$source_file" | cmp - "$patch_file"

build_name="inkling2-sycl-build-$$"
check_name="inkling2-sycl-check-$$"
owner="$(id -u):$(id -g)"
cleanup() {
    docker stop --time 5 "$build_name" "$check_name" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$mode" == exl3 ]]; then
    mkdir -p .work/python
    ln -sfn ../trellis-serve/xpu/exl3xpu .work/python/inkling_trellis
    tile_rows=${INKLING2_EXL3_TILE_ROWS:-16}
    case "$tile_rows" in 16|32|48) ;; *) echo 'Use 16, 32 or 48 tile rows.' >&2; exit 2;; esac
    lib_name=inkling2-exl3-moe.so
    if [[ "$tile_rows" != 16 ]]; then
        lib_name="inkling2-exl3-moe-mb$tile_rows.so"
    fi
    build_lib="/build/$lib_name"
    library=${INKLING2_EXL3_TEST_LIB:-"/lab/.work/build/$lib_name"}
    if [[ ${INKLING2_EXL3_BUILD:-1} == 1 ]]; then
        # The pinned vendor wrapper filters compiler output and then checks
        # the output path. Only a fresh compiler artifact can pass that check.
        rm -f -- "$output/$lib_name.new"
        patch --silent --batch --fuzz=0 --output="$output/exl3_moe_single.sycl" "$vendor/$source_file" \
            "$PWD/patches/trellis-inkling-exl3-single-row.patch"
        timeout --kill-after=10s 8m docker run --rm --name "$build_name" --network none \
            --cpus 4 --memory 6g --user "$owner" \
            --mount "type=bind,source=$vendor,target=/opt/trellis-serve,readonly" \
            --mount "type=bind,source=$output/exl3_moe_single.sycl,target=/opt/trellis-serve/$source_file,readonly" \
            --mount "type=bind,source=$output,target=/build" \
            --env "EXL3_MOE_OUT=$build_lib.new" --env "EXL3_MOE_FLAGS=-DEXL3_INKLING2 -DEXL3_MOE_PREFILL_MB=$tile_rows" \
            --env "CXX_STD=${INKLING2_CXX_STD:-c++17}" --entrypoint /bin/bash "$builder" /opt/trellis-serve/xpu/scripts/build_moe.sh \
            >"$output/exl3-compile.log" 2>&1
        if [[ -z ${INKLING2_EXL3_TEST_LIB:-} ]]; then
            library="/lab/.work/build/$lib_name.new"
        fi
    fi
    if [[ ${INKLING2_EXL3_TEST:-1} == 1 ]]; then
        timeout --kill-after=10s 3m docker run --rm --name "$check_name" --network none \
            --cpus 4 --memory 5g --device "$(readlink -f -- "$render_node")" \
            --mount "type=bind,source=$PWD,target=/lab,readonly" \
            --env PYTHONPATH=/lab:/lab/.work/python:/lab/.work/trellis-serve/core/src \
            --env PYTHONDONTWRITEBYTECODE=1 --env "EXL3_MOE_LIB=$library" --env SYCL_CACHE_PERSISTENT=0 \
            --entrypoint /opt/venv/bin/python "$runtime" /lab/tests/check_exl3_inkling.py /lab/models/exl3 "${@:2}" \
            | tee "$output/exl3-parity.log"
    fi
    if [[ ${INKLING2_EXL3_BUILD:-1} == 1 ]]; then
        mv -- "$output/$lib_name.new" "$output/$lib_name"
    fi
    exit
fi

if ! timeout --kill-after=10s 5m docker run --rm --name "$build_name" --network none \
    --cpus 3 --memory 4g --user "$owner" \
    --mount "type=bind,source=$vendor,target=/opt/trellis-serve,readonly" \
    --mount "type=bind,source=$output,target=/build" \
    --env INKLING2_OUT=/build/inkling2-vllm.so.new --env "CXX_STD=${INKLING2_CXX_STD:-c++17}" --entrypoint /bin/bash "$builder" \
    /opt/trellis-serve/xpu/scripts/build_inkling2.sh >"$output/compile.log" 2>&1; then
    tail -60 "$output/compile.log" >&2
    exit 1
fi

library=/lab/.work/build/inkling2-vllm.so.new
timeout --kill-after=10s 3m docker run --rm --name "$check_name" --network none \
    --cpus 3 --memory 3g --device "$(readlink -f -- "$render_node")" \
    --mount "type=bind,source=$PWD,target=/lab,readonly" \
    --env PYTHONPATH=/lab:/lab/.work/trellis-serve/xpu --env PYTHONDONTWRITEBYTECODE=1 \
    --env INKLING2_LIB="$library" --env SYCL_CACHE_PERSISTENT=0 \
    --entrypoint /opt/venv/bin/python "$runtime" /lab/tests/check_sycl.py "$library" \
    | tee "$output/parity.log"
mv -- "$output/inkling2-vllm.so.new" "$output/inkling2-vllm.so"
