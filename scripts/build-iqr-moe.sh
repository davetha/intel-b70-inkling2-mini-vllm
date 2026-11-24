#!/usr/bin/env bash
# Build .work/build/inkling2-iqr-moe.so (torch library iqr_moe: IQR 4-bit routed experts, decode + prefill) for the
# B70 (AOT bmg_g31). Two translation units from the same source: IQR_PART=1 (host, decode, align, combine; default
# options) and IQR_PART=2 (prefill GEMMs, -fno-sycl-rdc, IGC -ffp-contract=off so the unrolled fp32 epilogue keeps
# separate mul/add, i.e. stays bit-identical to the non-unrolled kernel). Extra icpx flags (e.g. -DIQR_RB_UNROLL=1)
# via IQR_MOE_FLAGS; output name via IQR_MOE_OUT.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
builder=${INKLING2_BUILDER:-local/inkling2-oneapi-034:2026.1.1}
out=${IQR_MOE_OUT:-inkling2-iqr-moe.so}
mkdir -p .work/build
docker run --rm --network none --cpus 16 --memory 16g --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$PWD,target=/lab" --env "IQR_MOE_FLAGS=${IQR_MOE_FLAGS:-}" --env "OUT=$out" \
    --entrypoint /bin/bash "$builder" -c '
    set -eo pipefail
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    torch=$(python3 -c "import os,torch; print(os.path.dirname(torch.__file__))")
    T=$(mktemp -d)
    C="icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 -O3 \
       -fno-fast-math -std=c++20 -Wno-deprecated-declarations -fPIC $IQR_MOE_FLAGS -I$torch/include -I$torch/include/torch/csrc/api/include"
    $C -DIQR_PART=1 -x c++ /lab/inkling_xpu/csrc/iqr_moe.sycl -c -o $T/p1.o
    $C -DIQR_PART=2 -fno-sycl-rdc -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -ffp-contract=off" \
        -x c++ /lab/inkling_xpu/csrc/iqr_moe.sycl -c -o $T/p2.o
    icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -fsycl-device-code-split=per_kernel -shared -fPIC $T/p1.o $T/p2.o \
        -L"$torch/lib" -Wl,-rpath,"$torch/lib" -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -o "/lab/.work/build/$OUT.new"
'
mv -- ".work/build/$out.new" ".work/build/$out"
ls -la ".work/build/$out"
