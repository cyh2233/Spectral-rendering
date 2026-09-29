# Building the CUDA / OptiX backend

The GPU backend runs the **same** integrator code as the CPU backend (`include/spectral/kernel/*`),
inside a single OptiX ray-generation program. OptiX only performs ray/triangle traversal
(one GAS per mesh, one IAS over instances, any-hit for alpha-masked foliage).

## Requirements
- NVIDIA driver supporting CUDA 12.8+ (Blackwell / RTX PRO 6000: driver R570 or newer)
- CUDA Toolkit >= 12.0 (12.8+ recommended for sm_120)
- OptiX SDK >= 8.0 (developed against OptiX 9.x headers)
- CMake >= 3.20

## Configure and build
```bash
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSPECTRAL_ENABLE_CUDA=ON -DOPTIX_ROOT=/path/to/NVIDIA-OptiX-SDK-9.x \
      -DCMAKE_CUDA_ARCHITECTURES=120      # Blackwell; default 86 (PTX JIT-compiled by the driver)
cmake --build build-gpu
ctest --test-dir build-gpu                 # CPU tests (GPU is exercised by the comparison below)
```
Windows (MSVC): same options with `-G "Visual Studio 17 2022"`; the default OptiX install path
`C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0` is searched automatically.

## Validate against the CPU backend
```bash
python scripts/compare_backends.py build-gpu/spectral_render examples/windshield/scene.json --spp 256
```
Mean per-band difference should be below 1 %; depth and segmentation AOVs should be identical.

## Selecting the backend
- JSON: `"render": {"backend": "cuda"}` (or `"auto"`: CUDA when available, else CPU)
- CLI: `spectral_render scene.json --backend cuda`

## What was verified without a GPU
This repository's CI container has no GPU and no nvcc. `scripts/check_cuda_compile.sh` compiles the
device programs to PTX with clang's CUDA mode and syntax-checks the host backend against the real
CUDA 12.9 and OptiX 9.1 headers. The first run on a GPU machine should still be treated as a
validation step (compare_backends.py).

## Memory
Film: `width * height * bands * 4` bytes on the device (1920x1080x125 ~ 1.04 GB), plus the
environment table (512x256xbands) and geometry. A 96 GB GPU holds full CARLA towns comfortably.
Per-thread path state is two dense spectra (2 x 516 B) plus temporaries in local memory.
