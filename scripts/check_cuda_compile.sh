#!/usr/bin/env bash
# Compile-checks the CUDA/OptiX backend on a machine WITHOUT nvcc or a GPU:
#  - device programs -> PTX with clang's CUDA mode,
#  - host backend with the system C++ compiler (-fsyntax-only).
# Needs: clang++ (>= 16), python3 venv with `pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12
#        nvidia-cuda-cccl-cu12 nvidia-curand-cu12`, and OptiX headers (git clone https://github.com/NVIDIA/optix-dev).
# Usage: scripts/check_cuda_compile.sh <site-packages/nvidia dir> <optix include dir>
set -euo pipefail
NV=${1:?path to site-packages/nvidia}
OPTIX_INC=${2:?path to OptiX include dir}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
CUDA="$WORK/cuda"
mkdir -p "$CUDA/include" "$CUDA/bin" "$CUDA/nvvm"
for d in cuda_runtime cuda_nvcc cuda_cccl curand; do cp -r --update=none "$NV/$d/include/"* "$CUDA/include/" 2>/dev/null || true; done
cp -r "$NV/cuda_nvcc/nvvm/"* "$CUDA/nvvm/"
cp "$NV/cuda_nvcc/bin/ptxas" "$CUDA/bin/"
clang++ -x cuda --cuda-path="$CUDA" --cuda-gpu-arch=sm_86 --cuda-device-only -S -O3 -std=c++17 -ffast-math \
  -Wno-unknown-cuda-version -I "$ROOT/include" -I "$OPTIX_INC" -DSPECTRAL_MAX_BANDS=128 \
  "$ROOT/src/backend/cuda/programs/device_programs.cu" -o "$WORK/device_programs.ptx"
grep -c "\.entry" "$WORK/device_programs.ptx" | xargs echo "PTX entry points:"
clang++ -x cuda --cuda-path="$CUDA" --cuda-gpu-arch=sm_86 --cuda-device-only -S -O3 -std=c++17 \
  -Wno-unknown-cuda-version -I "$ROOT/include" -DSPECTRAL_MAX_BANDS=128 \
  "$ROOT/src/backend/cuda/preview_kernels.cu" -o "$WORK/preview_kernels.ptx"
grep -c "\.entry" "$WORK/preview_kernels.ptx" | xargs echo "preview kernels:"
g++ -std=c++20 -fsyntax-only -Wall -Wextra -Wno-unused-parameter -I "$ROOT/include" -I "$ROOT/third_party" \
  -I "$CUDA/include" -I "$OPTIX_INC" -DSPECTRAL_HAS_CUDA=1 -DSPECTRAL_MAX_BANDS=128 \
  "$ROOT/src/backend/cuda/cuda_backend.cpp"
echo "host backend: OK"
rm -rf "$WORK"
