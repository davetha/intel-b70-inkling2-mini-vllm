#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:44' +%s)}
(( $(date +%s) + 150 < deadline ))
builder=${INKLING2_BUILDER:-sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca}
name="inkling2-w8a8-build-$$"
mkdir -p .work/build
trap 'docker stop --timeout 5 "$name" >/dev/null 2>&1 || true' EXIT
timeout --kill-after=10s 2m docker run --rm --name "$name" --network none --cpus 3 --memory 5g \
    --user "$(id -u):$(id -g)" --mount "type=bind,source=$PWD,target=/lab" \
    --env "CXX_STD=${INKLING2_CXX_STD:-c++17}" --entrypoint /bin/bash "$builder" -c '
    set -eo pipefail
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    torch=$(python3 -c "import os,torch; print(os.path.dirname(torch.__file__))")
    icpx -fsycl -fsycl-targets=spir64 -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 \
        -O3 -std=${CXX_STD:-c++17} -shared -fPIC -I"$torch/include" -I"$torch/include/torch/csrc/api/include" \
        -x c++ /lab/inkling_xpu/csrc/dense_w8a8.sycl -L"$torch/lib" \
        -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -o /lab/.work/build/inkling2-dense-w8a8.so.new
'
mv -- .work/build/inkling2-dense-w8a8.so.new .work/build/inkling2-dense-w8a8.so
