#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
release=b12588-mix-c0c3de8
checksum=92ee8439f9ae51a40f15c168ac64e1d657bf8b08d301c96f8fb2d58625573607
builder=${INKLING2_BUILDER:-sha256:ae6950731b3c031f812a95c0eb615a572239426440bf67fd1b3c9c9cbfce9eca}
runtime=${INKLING2_RUNTIME:-sha256:131dd9e11809de2f3136547618f130ee238f0b90cd5718567bcdd0dad5360e19}
work=${INKLING2_BUILD_DIR:-"$PWD/.work/llama-unsloth"}
jobs=${INKLING2_BUILD_JOBS:-7}
patches=("$PWD/patches/llama-sycl-banded-attention.patch")
if [[ ${INKLING2_BUILD_DSPARK_DEVICE:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-dspark-device-features.patch")
fi
if [[ ${INKLING2_BUILD_BANDED_DECODE:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-sycl-banded-decode.patch")
fi
if [[ ${INKLING2_BUILD_NVFP4_PREFILL:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-sycl-nvfp4-grouped-prefill.patch")
fi
if [[ ${INKLING2_BUILD_SWA_PREFILL:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-sycl-swa-prefill-bias.patch")
fi
if [[ ${INKLING2_BUILD_MASK_SKIP:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-sycl-attention-skip-masked-tiles.patch")
fi
if [[ ${INKLING2_BUILD_ESIMD_PREFILL:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-sycl-inkling-esimd-prefill.patch")
fi
if [[ ${INKLING2_BUILD_DSPARK_TOP1:-0} == 1 ]]; then
    patches+=("$PWD/patches/llama-dspark-sycl-top1.patch")
fi
deadline=${INKLING2_DEADLINE_EPOCH:-$(TZ=Europe/Berlin date -d 'today 22:45' +%s)}
(( $(date +%s) + 38 * 60 < deadline ))

docker image inspect "$builder" "$runtime" >/dev/null
mkdir -p "$work" "$work/deps" "$work/build"
exec 9>"$work/build.lock"
flock -n 9

archive="$work/source.tar.gz"
if [[ ! -s "$archive" ]]; then
    curl --fail --location --silent --show-error --retry 2 --max-time 90 \
        "https://github.com/unslothai/llama.cpp/releases/download/$release/llama.cpp-source-$release.tar.gz" \
        -o "$archive.part"
    printf '%s  %s\n' "$checksum" "$archive.part" | sha256sum --check --status
    mv -- "$archive.part" "$archive"
fi
printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status
if [[ ! -f "$work/source/CMakeLists.txt" ]]; then
    mkdir -p "$work/source"
    tar -xzf "$archive" --strip-components=1 -C "$work/source"
fi
check_dir=$(mktemp -d)
trap 'rm -r -- "$check_dir"' EXIT
while IFS= read -r path; do
    if [[ -f "$work/source/$path" ]]; then
        install -D -m 600 "$work/source/$path" "$check_dir/$path"
    fi
done < <(sed -n 's|^+++ b/\([^[:space:]]*\).*|\1|p' "${patches[@]}" | sort -u)
already_patched=true
for (( index=${#patches[@]} - 1; index >= 0; index-- )); do
    if ! patch --silent --batch --reverse --fuzz=0 -d "$check_dir" -p1 < "${patches[index]}" >/dev/null 2>&1; then
        already_patched=false
        break
    fi
done
if ! "$already_patched"; then
    for patch_file in "${patches[@]}"; do
        if ! patch --dry-run --silent --batch --reverse --fuzz=0 -d "$work/source" -p1 < "$patch_file" >/dev/null 2>&1; then
            patch --batch --forward --fuzz=0 -d "$work/source" -p1 < "$patch_file"
        fi
    done
fi
rm -r -- "$check_dir"
trap - EXIT

if [[ ! -s "$work/deps/b70/lib/libdnnl.so.3" ]]; then
    docker run --rm --network none --entrypoint tar "$runtime" -C /opt -cf - b70 \
        | tar -C "$work/deps" -xf -
fi
mkdir -p "$work/deps/sycl-runtime" "$work/deps/mkl"
if [[ ! -s "$work/deps/sycl-runtime/libsycl.so.9" ]]; then
    docker run --rm --network none --entrypoint cat "$runtime" /opt/venv/lib/libsycl.so.9 \
        > "$work/deps/sycl-runtime/libsycl.so.9"
fi
ln -sf libsycl.so.9 "$work/deps/sycl-runtime/libsycl.so"
python3 - "$work/deps/mkl" <<'PY'
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

root = Path(sys.argv[1])
packages = {
    "mkl-devel": "98c54ad5c82681cc3f516495c9c8ee9135b5d54e8f0ebbcb1ec65f09fc0a9a87",
    "mkl-include": "2a0f5fe9f10f5b3d335fec6c3a3a6f7ed76900e04b7f7c92ebeb93a017cb0e6d",
    "onemkl-sycl-include": "91f2024ea0ecae4894ce4eccf6b16fd89e477e3b05c3c87124261f2399ff1844",
}
for package, sha in packages.items():
    receipt = root / (package + ".sha256")
    if receipt.is_file() and receipt.read_text() == sha:
        continue
    metadata = f"https://pypi.org/pypi/{package}/2026.0.0/json"
    with urllib.request.urlopen(metadata, timeout=15) as response:
        release = json.load(response)
    asset = next(item for item in release["urls"] if item["digests"]["sha256"] == sha)
    with urllib.request.urlopen(asset["url"], timeout=15) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"Wrong checksum for {package}")
    with zipfile.ZipFile(io.BytesIO(data)) as wheel:
        for name in wheel.namelist():
            if ".data/data/" in name and not name.endswith("/"):
                path = root / name.split(".data/data/", 1)[1]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(wheel.read(name))
    receipt.write_text(sha)
PY

name="inkling2-llama-build-$$"
trap 'docker stop --timeout 5 "$name" >/dev/null 2>&1 || true' EXIT
timeout --kill-after=15s 35m docker run --rm --name "$name" --network none \
    --memory 15g --cpus "$jobs" --env "BUILD_JOBS=$jobs" --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$work,target=/lab" --entrypoint bash "$builder" -c '
    set -eo pipefail
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
    for file in /opt/venv/lib/libmkl*.so.*; do
        name=$(basename "$file")
        ln -sf "$file" /lab/deps/mkl/lib/"${name%%.so.*}.so"
    done
    cmake -S /lab/source -B /lab/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
        "-DCMAKE_BUILD_RPATH=/lab/deps/b70/lib;/opt/venv/lib" \
        "-DCMAKE_EXE_LINKER_FLAGS=-L/lab/deps/sycl-runtime -Wl,-rpath-link,/lab/deps/sycl-runtime -Wl,--no-as-needed -lsycl -Wl,--as-needed" \
        -DMKL_ROOT=/lab/deps/mkl -DMKL_DIR=/lab/deps/mkl/lib/cmake/mkl -DMKL_LINK=dynamic -DMKL_INTERFACE=lp64 \
        -DDNNL_DIR=/lab/deps/b70/lib/cmake/dnnl -DGGML_SYCL=ON -DGGML_SYCL_F16=ON \
        -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
    cmake --build /lab/build --target llama-server llama-bench llama-quantize -j "$BUILD_JOBS"
' > "$work/build-sycl-baseline.log" 2>&1
