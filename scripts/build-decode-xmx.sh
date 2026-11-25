#!/usr/bin/env bash
# Build .work/build/inkling2-decode-xmx.so (torch library inkling2_dxmx: the decode XMX attention kernels,
# inkling_xpu/csrc/decode_xmx.sycl) for the B70: AOT bmg_g31, 256-GRF register file (the JIT 128-GRF build spills),
# same -O3 -ffast-math as the vendored inkling2_xpu build. Output name via DECODE_XMX_OUT.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
builder=${INKLING2_BUILDER:-local/inkling2-oneapi-034:2026.1.1}
out=${DECODE_XMX_OUT:-inkling2-decode-xmx.so}
mkdir -p .work/build
docker run --rm --network none --cpus 16 --memory 24g --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$PWD,target=/lab" --env "OUT=$out" --entrypoint /bin/bash "$builder" -c '
    set -eo pipefail
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    torch=$(python3 -c "import os,torch; print(os.path.dirname(torch.__file__))")
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 -O3 \
        -ffast-math -std=c++20 -fPIC -shared -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -ze-opt-large-register-file" \
        -I$torch/include -I$torch/include/torch/csrc/api/include -x c++ /lab/inkling_xpu/csrc/decode_xmx.sycl -x none \
        -L"$torch/lib" -Wl,-rpath,"$torch/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -o "/lab/.work/build/$OUT.new"
'
mv -- ".work/build/$out.new" ".work/build/$out"
ls -la ".work/build/$out"
