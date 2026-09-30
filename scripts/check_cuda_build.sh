#!/usr/bin/env bash
# Real CMake/nvcc build of the CUDA backend on a machine WITHOUT a GPU or CUDA toolkit:
# assembles a toolkit from NVIDIA's pip wheels (nvcc, cudart, cccl, crt, nvvm), adds an empty libcuda stub
# for linking, configures with -DSPECTRAL_ENABLE_CUDA=ON (OptiX headers fetched), builds, and runs the tests on
# the CPU fallback. Needs python3 with pip, cmake, ninja, g++ and network access to PyPI and GitHub.
# Usage: scripts/check_cuda_build.sh [work dir (default /tmp/spectral-cuda-check)]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${1:-/tmp/spectral-cuda-check}
PY=${PYTHON:-python3}
mkdir -p "$WORK"
if [ ! -x "$WORK/pkgs/nvidia/cu13/bin/nvcc" ]; then
  "$PY" -m pip install -q --target "$WORK/pkgs" nvidia-cuda-nvcc nvidia-cuda-runtime nvidia-cuda-cccl \
    nvidia-cuda-crt nvidia-nvvm
fi
CUDA="$WORK/pkgs/nvidia/cu13"
ln -sf libcudart.so.13 "$CUDA/lib/libcudart.so"
mkdir -p "$CUDA/lib/stubs"
[ -f "$CUDA/lib/stubs/libcuda.so" ] || { echo "" > "$WORK/empty.c"; gcc -shared -fPIC -o "$CUDA/lib/stubs/libcuda.so" "$WORK/empty.c"; }
CUDACXX="$CUDA/bin/nvcc" cmake -S "$ROOT" -B "$WORK/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_ENABLE_CUDA=ON
cmake --build "$WORK/build"
PTX=$(find "$WORK/build" -name device_programs.ptx | head -1)
grep -q "\.visible \.const .* params\[" "$PTX" && echo "PTX: params is .visible"
echo "PTX entry points: $(grep -c '\.entry' "$PTX")"
LD_LIBRARY_PATH="$CUDA/lib" ctest --test-dir "$WORK/build" --output-on-failure
echo "CUDA build check: OK"
