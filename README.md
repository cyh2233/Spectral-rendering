# Spectral Rendering for Camera Simulation

카메라 파이프라인(센서 시뮬레이션)에 입력할 **분광(spectral) 이미지**를 만드는 물리 기반 렌더러입니다.

- **Dense-band 분광 path tracing** (hero wavelength 미사용): 한 경로가 모든 밴드(기본 380–1000 nm, 5 nm, 125 밴드)를 동시에 운반합니다.
- **C++ 라이브러리 `libspectral` + CLI**, CPU(OpenMP) 백엔드와 **CUDA/OptiX** 백엔드가 같은 커널 코드를 공유합니다.
- **입력**: glTF 에셋 + 장면 JSON (Blender 애드온, CARLA 변환기 제공). **출력**: 멀티스펙트럴 EXR / NPZ + depth + semantic ID.
- **재질**: glTF PBR → Jakob–Hanika RGB 업리프팅, 측정 분광 데이터로 오버라이드, 윈드실드용 유리(분광 Fresnel, 흡수, 틴트/IR-cut).
- **광원**: Prague Sky Model(SWIR 데이터셋), 외부 스카이 플러그인(C ABI, Connor 모듈 연결용), 점/스팟/면 광원(CIE LED·HPS 등 분광), HDR 환경맵.
- **광학 단계 (`optics/`, Python)**: 블랙박스 렌즈의 벤더 데이터(PSF/MTF/왜곡/주변광량), RayOptics 처방(.zmx/.seq), **윈드실드 파면 모델**로 PSF를 만들고 렌더 결과에 적용합니다.

> English documentation below and in `docs/`.

## Status
| Component | State |
|---|---|
| CPU renderer, scene/glTF loading, EXR/NPZ | Implemented, 55 C++ test cases passing |
| Live session (incremental updates, preview) + Python module | Implemented, tested on CPU (moved/removed instances bit-identical to a fresh scene) |
| CUDA/OptiX backend | Implemented; compiled to PTX and syntax-checked against CUDA 12.9 / OptiX 9.1 headers here, **not yet run on a GPU** (see `docs/gpu_build.md`) |
| Prague sky | Integrated and compiled; **not run against a dataset** (download blocked here) — gated test available |
| Optics stage | Implemented, 19 Python tests passing (incl. renderer → optics end-to-end) |
| Blender add-on | Implemented, tested headless with `bpy` 4.2 (export → render → segmentation/colour checks) |
| CARLA converter | Implemented, offline tests only (no CARLA server here) |
| CARLA live viewer | Implemented, tested against a fake CARLA API with headless pygame; **not yet run against a CARLA server** |
| Material spectra | `approx_*` files are **placeholders, not measurements** — replace with measured data |

## Build
```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build          # first run generates the RGB->spectrum table (~8 s, cached)
```
GPU build: `-DSPECTRAL_ENABLE_CUDA=ON -DOPTIX_ROOT=/path/to/OptiX-SDK -DCMAKE_CUDA_ARCHITECTURES=120`
(see `docs/gpu_build.md`). All dependencies are vendored in `third_party/` (no network needed).

Python tools:
```bash
pip install -e optics[rayoptics,test]
python -m pytest optics/tests tools/carla_export
```

## Quick start
```bash
build/spectral_render examples/windshield/scene.json          # writes examples/windshield/out/*
spectral-optics windshield --focal-length 6 --f-number 1.8 --fov 60 34 -o ws.optics.npz
spectral-optics apply examples/windshield/out/windshield.npz ws.optics.npz -o optical.npz --irradiance
```
```python
import numpy as np
z = np.load("examples/windshield/out/windshield.npz")
z["radiance"].shape   # (360, 640, 125)  W/(m^2 sr nm)
z["wavelengths"]      # 380 ... 1000
z["depth"], z["seg_id"]
```
C++ API (`#include <spectral/api.h>`, CMake `find_package(spectral)` → `spectral::spectral`):
```cpp
spectral::Renderer r;
r.load_scene_file("scene.json");
spectral::SpectralImage img = r.render();       // img.radiance[(y*W + x)*bands + b]
spectral::Renderer::write_outputs(img, r.scene().output);
```
Live session from Python (`-DSPECTRAL_BUILD_PYTHON=ON`, module in `build/python`):
```python
from spectral_renderer import Session
s = Session("examples/windshield/scene.json", backend="cuda")
h = s.spawn({"primitive": "box", "size": [1, 1, 1], "material": {"base_color": [0.6, 0.05, 0.05]},
             "transform": M, "seg_id": 14})
s.set_transform(h, M2); s.reset(); s.render(16)
rgb8 = s.preview("srgb")          # HxWx3 uint8, computed on the device
cube = s.image()["radiance"]      # HxWxB float32
```
Online CARLA: `PYTHONPATH=build/python python tools/carla_live/spectral_live.py --config my_live.json`
(see `docs/carla_live.md`).

## Documentation
- `docs/architecture.md` — design, dense-band transport, code layout, tests
- `docs/scene_json_schema.md` — scene format reference
- `docs/windshield.md` — windshield modelling split between renderer and optics
- `docs/optics_stage.md` — optics packages, vendor data for black-box lenses, applying optics
- `docs/blender_export.md`, `docs/carla_export.md` — scene sources
- `docs/carla_live.md` — online CARLA viewer (CARLA RGB | spectral render, live/quality modes)
- `docs/sky_plugin.md` — Prague sky, plugin ABI
- `docs/gpu_build.md` — CUDA/OptiX build and validation
- `data/spectra/materials/README.md` — material spectra provenance

## Known limitations
- NIR (>780 nm) of RGB-textured materials is an assumption (held constant); use measured spectra
  for NIR-relevant materials.
- No participating media (fog/rain) yet; the integrator keeps a transmittance hook.
- Rough dielectric transmission is not implemented (glass is smooth or thin).
- Pinhole camera only; lens effects come from the optics stage (object at infinity).
