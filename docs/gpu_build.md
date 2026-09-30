# Building the CUDA / OptiX backend

The GPU backend runs the **same** integrator code as the CPU backend (`include/spectral/kernel/*`),
inside a single OptiX ray-generation program. OptiX only performs ray/triangle traversal
(one GAS per mesh, one IAS over instances, any-hit for alpha-masked foliage).

## Requirements
- NVIDIA driver R570 or newer (Blackwell / RTX PRO 6000); `nvidia-smi` must work
- CUDA Toolkit >= 12.0 (12.8+ for native sm_120 code); the driver must be at least as new as the toolkit
  (`nvidia-smi` "CUDA Version" >= `nvcc --version`), otherwise the PTX cannot be loaded
- CMake >= 3.20
- OptiX: nothing to install. The OptiX runtime is part of the driver; the headers are fetched at configure
  time from github.com/NVIDIA/optix-dev (tag `SPECTRAL_OPTIX_TAG`, default `v9.0.0`, which needs R570+).
  Offline machines: `-DOPTIX_ROOT=/path/to/NVIDIA-OptiX-SDK-9.x` (or `-DSPECTRAL_FETCH_OPTIX=OFF`).

## Configure and build
```bash
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPECTRAL_ENABLE_CUDA=ON
cmake --build build-gpu
ctest --test-dir build-gpu                 # CPU tests (GPU is exercised by the comparison below)
```
`scripts/setup.sh` does all of this (plus the Python module and tests). The configure log shows
`CUDA architectures: 120` (read from `nvidia-smi --query-gpu=compute_cap`; override with
`-DCMAKE_CUDA_ARCHITECTURES=...`) and `OptiX headers: ...`. With nvcc older than 12.8 on Blackwell the build
falls back to sm_90 PTX, which the driver JIT-compiles.
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
The development container has no GPU. `scripts/check_cuda_build.sh` assembles a CUDA toolkit from NVIDIA's
pip wheels (real nvcc 13.x) and runs the real CMake build with `-DSPECTRAL_ENABLE_CUDA=ON` (OptiX headers
fetched), then runs the tests on the CPU fallback: device programs, preview kernels and the host backend all
compile and link, and the PTX exports the `params` launch-parameter symbol as `.visible`.
`scripts/check_cuda_compile.sh` is an older clang-based check (PTX + host syntax only).
The first run on a GPU machine is still the real validation step (compare_backends.py).

## Memory
Film: `width * height * bands * 4` bytes on the device (1920x1080x125 ~ 1.04 GB), plus the
environment table (512x256xbands) and geometry. A 96 GB GPU holds full CARLA towns comfortably.
Per-thread path state is two dense spectra (2 x 516 B) plus temporaries in local memory.
