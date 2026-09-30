# Spectral Rendering for Camera Simulation

카메라 파이프라인(센서 시뮬레이션)에 입력할 **분광(spectral) 이미지**를 만드는 물리 기반 path tracer 입니다.
이 문서는 **서버에서 그대로 따라 치면 되는 명령어** 위주의 매뉴얼입니다. 설계/포맷 상세는 `docs/` 를 보세요.

- Dense-band 분광 path tracing (hero wavelength 미사용): 한 경로가 380–1000 nm 전 밴드를 동시에 운반
- CPU(OpenMP) + CUDA/OptiX 백엔드 (같은 커널 코드), C++ 라이브러리 / CLI / Python 모듈
- 출력: 멀티스펙트럴 EXR/NPZ (radiance W/(m²·sr·nm)) + depth + semantic ID
- 입력: glTF + 장면 JSON (Blender 애드온, CARLA 오프라인 변환기, **CARLA 온라인 뷰어**)
- 광학 단계(`optics/`): 블랙박스 렌즈 벤더 데이터 / RayOptics 처방 / 윈드실드 → PSF·왜곡·주변광량 적용

```
Blender / CARLA ──► scene.json + .glb ──► spectral_render (CPU|CUDA) ──► cube.npz/.exr ──► spectral-optics ──► 센서 입력
CARLA server ─────────────────────────► spectral_live.py (Session, 실시간) ──► 창: CARLA RGB | spectral
```

---

## 0. 3줄 요약 (GPU 서버)

```bash
git clone https://github.com/cyh2233/Spectral-rendering.git && cd Spectral-rendering
scripts/setup.sh                 # 사전조건 확인 → venv → 빌드(CUDA 자동) → 테스트 → env.sh
source env.sh && spectral_render examples/windshield/scene.json --backend cuda
```

이미 받아 둔 저장소라면:
```bash
cd Spectral-rendering
git pull
scripts/setup.sh                 # 다시 실행해도 됨 (바뀐 것만 다시 빌드)
```

---

## 1. 서버 요구사항 확인

| 항목 | 버전 | 확인 명령 |
|---|---|---|
| Ubuntu | 22.04 권장 (20.04/24.04 가능) | `lsb_release -a` |
| NVIDIA 드라이버 | R570 이상 (Blackwell) | `nvidia-smi` |
| CUDA Toolkit | 12.8 이상 권장 (sm_120 네이티브) | `nvcc --version` |
| CMake | 3.20 이상 | `cmake --version` |
| g++ | 11 이상 (C++20) | `g++ --version` |
| Python | 3.10 (CARLA 0.9.15/0.9.16 공통) | `python3.10 --version` |
| OptiX | **설치 불필요** — 런타임은 드라이버에 포함, 헤더는 빌드 시 자동 다운로드 | |

드라이버의 "CUDA Version"(nvidia-smi 오른쪽 위)이 `nvcc` 버전 이상이어야 합니다.

부족한 패키지 설치:
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git python3.10 python3.10-venv python3.10-dev
# CUDA Toolkit이 없으면 (예: 12.8, NVIDIA 저장소 사용)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-12-8
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
```
Ubuntu 22.04 에 python3.10 이 기본입니다. 24.04 라면 `sudo add-apt-repository ppa:deadsnakes/ppa` 후 설치하세요.

---

## 2. 설치: `scripts/setup.sh`

```bash
scripts/setup.sh                          # 기본: CUDA 자동 감지, venv=.venv, build=build, 테스트 실행
scripts/setup.sh --with-carla 0.9.15      # CARLA Python client 도 venv 에 설치 (0.9.16 도 가능)
scripts/setup.sh --cuda off               # CPU 전용
scripts/setup.sh --no-tests -j 32         # 테스트 생략, 병렬 빌드 수
scripts/setup.sh -- -DSPECTRAL_OPTIX_TAG=v9.1.0   # '--' 뒤는 CMake 인자 그대로 전달
scripts/setup.sh --help
```

하는 일:
1. 사전조건 검사 (없으면 설치할 apt 명령을 출력하고 중단)
2. `.venv` 생성 + `numpy pygame pytest` + 광학 패키지(`spectral-optics`, RayOptics) (+ `carla`)
3. CMake 구성: Release, Python 모듈 ON, CUDA ON/OFF, GPU 아키텍처 자동(Blackwell → `120`), OptiX 헤더 자동 다운로드
4. 빌드 → `build/spectral_render`, `build/python/spectral_renderer`
5. `env.sh` 생성 (venv 활성화, `PATH`, `PYTHONPATH`, `CARLA_ROOT`)
6. 테스트: C++(ctest), Python(session, CARLA live(가짜 CARLA), CARLA export, optics), GPU가 있으면 CPU↔GPU 비교

정상 출력 끝부분:
```
  -- CUDA architectures: 120
  -- OptiX headers: .../build/_deps/optix_dev-src/include
100% tests passed, 0 tests failed out of 1
........................ passed
  GPU vs CPU comparison (examples/windshield, 64 spp):
image-mean per-band relative difference: max ..., mean <1%
Setup complete.
```

새 터미널마다:
```bash
cd Spectral-rendering && source env.sh
```

### 2.1 수동 빌드 (스크립트 없이)
```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install numpy pygame pytest -e "optics[rayoptics,test]"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSPECTRAL_ENABLE_CUDA=ON -DSPECTRAL_BUILD_PYTHON=ON -DPython3_EXECUTABLE=$(which python)
cmake --build build -j
export PYTHONPATH=$PWD/build/python:$PWD/tools/carla_live
```

| CMake 옵션 | 기본값 | 설명 |
|---|---|---|
| `SPECTRAL_ENABLE_CUDA` | OFF (setup.sh 는 자동) | CUDA/OptiX 백엔드 |
| `SPECTRAL_BUILD_PYTHON` | OFF (setup.sh 는 ON) | Python 모듈 `spectral_renderer` |
| `CMAKE_CUDA_ARCHITECTURES` | nvidia-smi 로 자동 (없으면 86) | 예: `120` (Blackwell) |
| `OPTIX_ROOT` | – | 로컬 OptiX SDK 경로 (지정 시 다운로드 안 함) |
| `SPECTRAL_FETCH_OPTIX` | ON | OptiX 헤더 자동 다운로드 (github.com/NVIDIA/optix-dev) |
| `SPECTRAL_OPTIX_TAG` | `v9.0.0` | 받을 헤더 버전 (9.0 = 드라이버 R570+) |
| `SPECTRAL_MAX_BANDS` | 128 | 경로당 최대 밴드 수 |
| `SPECTRAL_ENABLE_PRAGUE_SKY` | ON | Prague sky model |
| `SPECTRAL_BUILD_TESTS` | ON | doctest 테스트 |
| `SPECTRAL_BUILD_SHARED` | OFF | libspectral 공유 라이브러리 |

인터넷이 막힌 서버: OptiX SDK 를 받아 두고 `-DOPTIX_ROOT=/path/NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64`.

---

## 3. 테스트 / GPU 검증

```bash
source env.sh
ctest --test-dir build --output-on-failure                     # C++ 55 test cases (첫 실행 시 RGB→스펙트럼 LUT 생성 ~8 s)
python -m pytest -q python/tests tools/carla_live/tests tools/carla_export optics/tests
python scripts/compare_backends.py build/spectral_render examples/windshield/scene.json --spp 256
```
`compare_backends.py` 기대값: 밴드별 평균 차이 1 % 미만, `depth identical: True; seg_id identical: True`.
GPU 백엔드는 이 저장소 개발 환경(GPU 없음)에서 **빌드·링크까지만 검증**되었으므로, 서버에서 이 비교를 꼭 한 번 실행하세요.

Prague sky 데이터셋 테스트 (데이터셋이 있을 때):
```bash
SPECTRAL_PRAGUE_DATASET=/data/PragueSkyModelDatasetSWIR.dat build/tests/spectral_tests
```

---

## 4. 오프라인 렌더 (CLI)

```bash
source env.sh
spectral_render examples/colorchecker/scene.json               # ColorChecker, D65
spectral_render examples/windshield/scene.json --backend cuda  # 윈드실드 너머 도로
spectral_render examples/night_street/scene.json               # 야간: 가로등/헤드라이트
```
결과는 각 예제의 `out/` (예: `examples/windshield/out/windshield.{npz,exr}`, `windshield_srgb.png`).

옵션:
```bash
spectral_render scene.json \
  --backend cuda|cpu|auto   --spp 256   --threads 0   --seed 1 \
  --resolution 1920x1080 \
  --out-npz out/frame.npz  --out-exr out/frame.exr  --preview out/frame.png  --quiet
```

결과 읽기 (Python):
```python
import numpy as np
z = np.load("examples/windshield/out/windshield.npz")
z["radiance"]     # (H, W, B) float32, W/(m^2 sr nm)
z["wavelengths"]  # (B,) nm, 기본 380..1000 step 5 → 125 밴드
z["depth"]        # (H, W) m, 카메라 광선 거리
z["seg_id"]       # (H, W) uint32, 하늘 = 0xFFFFFFFF
z["spp"], z["metadata"]
```
EXR: 밴드 채널 `L_0380` … `L_1000` + `depth`(float) + `seg_id`(uint) (OpenEXR / OpenImageIO 로 읽기).

---

## 5. 장면 만들기

### 5.1 장면 JSON (최소 예)
```json
{
  "camera": {"resolution": [1920, 1080], "fov_deg": 60, "position": [0, 1.4, 0], "look_at": [0, 1.2, -10]},
  "assets": {"town": {"path": "town.glb"}},
  "instances": [{"asset": "town", "seg_id_by_material": {"(?i)road": 7}}],
  "materials": {"use_default_overrides": true},
  "lights": [{"type": "sky", "model": "prague", "dataset": "/data/PragueSkyModelDatasetSWIR.dat",
              "sun": {"elevation_deg": 35, "azimuth_deg": 120}, "visibility_km": 40}],
  "render": {"spp": 256, "max_depth": 6, "bands": {"min": 380, "max": 1000, "step": 5}, "backend": "auto"},
  "output": {"npz": "out/frame.npz", "exr": "out/frame.exr", "preview_png": "out/frame.png"}
}
```
전체 키: `docs/scene_json_schema.md`. 재질 측정 스펙트럼은 `data/spectra/` (현재 `approx_*` 는 **측정값이 아닌 자리표시자** → 실측 CSV 로 교체 권장, `data/spectra/materials/README.md`).

### 5.2 Blender → 장면
```bash
cd tools/blender_addon && zip -r ~/spectral_exporter.zip spectral_exporter && cd -
# Blender: Edit › Preferences › Add-ons › Install… › ~/spectral_exporter.zip → "Spectral Renderer Exporter" 체크
# File › Export › Spectral Scene (.json)  →  scene.json + scene.glb
spectral_render scene.json --backend cuda
```
헤드리스 검증: `pip install bpy==4.2.0 && python tools/blender_addon/test_export_blender.py build/spectral_render /tmp/bl_out`
(재질 커스텀 속성 `spectral_*`, seg_id 등: `docs/blender_export.md`)

### 5.3 CARLA → 장면 (오프라인, 프레임 단위)
```bash
cp tools/carla_export/asset_mapping.example.json my_mapping.json   # 타운/차량 glTF 경로 지정
python tools/carla_export/carla_to_scene.py --mapping my_mapping.json --out frames/f000123.json \
       --host localhost --port 2000 --headlights --dump-snapshot frames/f000123.snap.json
spectral_render frames/f000123.json
# CARLA 없이 저장된 스냅샷에서 다시 만들기
python tools/carla_export/carla_to_scene.py --mapping my_mapping.json --snapshot frames/f000123.snap.json --out f.json
```
(타운 glTF 추출 방법: `docs/carla_export.md`)

### 5.4 하늘 / 광원
| 종류 | JSON |
|---|---|
| 간단 하늘 | `{"type":"sky","model":"simple","sky_radiance":{"type":"cie","name":"D65","scale":0.0005},"sun":{...}}` |
| Prague | `{"type":"sky","model":"prague","dataset":"/data/PragueSkyModelDatasetSWIR.dat","sun":{...},"visibility_km":40}` |
| 플러그인 (Connor 모듈) | `{"type":"sky","model":"plugin","library":"/path/libsky.so","options":{...},"sun":{...}}` |
| 헤드라이트/가로등 | `{"type":"spot","position":[..],"direction":[..],"cone_inner_deg":12,"cone_outer_deg":30,"emission":{"type":"library","name":"cie_led_b5","photometric":8000}}` |
| 환경맵 | `{"type":"envmap","path":"sky.exr"}` |

Prague 데이터셋: Prague Sky Model 공개 저장소(PragueSkyModel)의 README 링크에서 **SWIR** 데이터셋(`PragueSkyModelDatasetSWIR.dat`, ~547 MB)을 받으세요 (가시광 전용 데이터셋은 760 nm 에서 끝남).
플러그인 C ABI: `docs/sky_plugin.md`.

---

## 6. 광학 단계 (`spectral-optics`)

렌더는 핀홀이고, 렌즈/윈드실드 효과는 이 단계에서 적용합니다.
```bash
source env.sh
# (a) 이상 렌즈 + 윈드실드
spectral-optics windshield --focal-length 6 --f-number 1.8 --fov 60 34 --rake 60 -o ws.optics.npz
# (b) 블랙박스 렌즈: 벤더 PSF/MTF/왜곡/주변광량 export 를 manifest 로 기술 (형식: docs/optics_stage.md)
spectral-optics vendor lens_manifest.json -o lens.optics.npz
# (c) 공개 처방(.zmx / CODE V .seq)을 RayOptics 로 추적
spectral-optics prescription lens.zmx --fields 0 10 20 30 --wavelengths 450 550 650 850 -o lens.optics.npz
# (d) 렌즈 × 윈드실드 결합
spectral-optics combine lens.optics.npz ws.optics.npz -o camera.optics.npz
# (e) 렌더 결과에 적용 (--irradiance: 센서면 조도로 변환)
spectral-optics apply examples/windshield/out/windshield.npz camera.optics.npz -o optical.npz --irradiance
```
렌더 해상도·FOV 를 센서 화소수·초점거리에 맞춰야 화소가 1:1 로 대응합니다 (`docs/optics_stage.md`, `docs/windshield.md`).
Zemax `.ZBB`(블랙박스)는 라이선스 없이 추적할 수 없으므로 벤더에 PSF/MTF/왜곡 export 를 요청해 (b) 로 씁니다.

---

## 7. CARLA 온라인 뷰어 (CARLA RGB | spectral 실시간)

### 7.1 CARLA 설치
```bash
source env.sh
scripts/install_carla.sh 0.9.15                  # ~/carla/CARLA_0.9.15 에 설치 + Python client + env.local.sh
# 다른 버전/경로, 또는 브라우저로 받은 tar 사용
scripts/install_carla.sh 0.9.16 /opt/carla
scripts/install_carla.sh 0.9.15 --from-tar ~/Downloads/CARLA_0.9.15.tar.gz --maps-tar ~/Downloads/AdditionalMaps_0.9.15.tar.gz
sudo apt install -y libomp5 xdg-user-dirs libvulkan1 vulkan-tools   # CARLA 런타임 라이브러리
vulkaninfo | head -5                                                # Vulkan 동작 확인
```
다운로드가 막히면 https://github.com/carla-simulator/carla/releases 에서 `CARLA_0.9.15.tar.gz` 를 받고 `--from-tar` 로 지정하세요.
`source env.sh` 를 다시 하면 `CARLA_ROOT` 가 설정됩니다.

### 7.2 실행 (서버 + 뷰어 한 번에)
```bash
source env.sh
scripts/run_carla_live.sh                                   # CARLA 창 + 뷰어 창
scripts/run_carla_live.sh --offscreen --quality Low         # CARLA 창 없이 (GPU 여유 ↑)
scripts/run_carla_live.sh --config my_live.json
scripts/run_carla_live.sh --no-server --host 10.0.0.5       # 이미 떠 있는/원격 CARLA 에 접속
scripts/run_carla_live.sh -- --frames 200                   # '--' 뒤는 spectral_live.py 인자
```
따로 실행하려면:
```bash
$CARLA_ROOT/CarlaUE4.sh -quality-level=Epic &               # 터미널 1 (또는 -RenderOffScreen)
python tools/carla_live/spectral_live.py --config tools/carla_live/config.example.json   # 터미널 2
```
원격 데스크톱/SSH 에서 창을 띄우려면 `ssh -X` 또는 VNC, 창 없이 저장만 하려면 `-- --headless`.

### 7.3 키
| 키 | 동작 |
|---|---|
| `M` | live ↔ quality 모드 (quality: CARLA 멈추고 `spp_quality` 까지 수렴 후 다음 tick) |
| `V` | 보기: sRGB → 단일 밴드 → NIR false color → depth → segmentation |
| `[` `]` | 단일 밴드 파장 이동 |
| `+` `-` | 노출 (EV) |
| `P` | 일시정지 (CARLA 정지, 렌더는 계속 누적) |
| `S` | 현재 큐브 저장 → `save_dir/snapshot_N.npz/.exr/_srgb.png` |
| `R` | 녹화: quality 모드에서 프레임마다 `frame_N.*` 저장 |
| `Esc` | 종료 (CARLA 설정 복원, 스폰한 액터 삭제) |

### 7.4 설정 파일 (`cp tools/carla_live/config.example.json my_live.json`)
| 키 | 예 | 설명 |
|---|---|---|
| `host`, `port` | `localhost`, `2000` | CARLA 서버 |
| `carla_town` | `Town10HD_Opt` | 다르면 로드 |
| `backend` | `cuda` | `cuda` / `cpu` / `auto` |
| `render_resolution` | `[960, 540]` | spectral 렌더 해상도 (live 속도 핵심) |
| `carla_resolution`, `panel_size` | `[1280, 720]`, `[960, 540]` | CARLA 카메라 / 창 패널 |
| `spp_live`, `spp_quality`, `spp_chunk` | `4`, `256`, `16` | 모드별 샘플 수 |
| `render.bands` | `{"min":380,"max":1000,"step":10}` | live 는 10 nm(63밴드), 최종은 5 nm |
| `render.max_depth` | `5` | 반사 횟수 |
| `fov`, `camera_mount` | `90`, `{"x":0.8,"z":1.3}` | 카메라 (CARLA 와 동일하게 적용) |
| `traffic`, `ego_blueprint`, `fixed_delta_seconds` | `20`, `vehicle.lincoln.mkz_2020`, `0.05` | 시뮬레이션 |
| `headlights`, `headlight` | `true`, 콘 각도 | 차량 라이트 상태 → 스펙트럴 스팟광 |
| `sky` | `{"model":"prague","dataset":...}` | 태양 방향은 CARLA weather 를 따라감 |
| `town`, `blueprints` | glTF 경로 | 있으면 glTF, 없으면 **proxy world** |
| `render_ego` | `false` | glTF ego 차량(보닛/윈드실드)을 렌더할지 |
| `materials` | 장면 JSON 형식 | proxy 재질을 측정 스펙트럼으로 교체 |
| `save_dir` | `carla_live_out` | 저장 폴더 |

**Proxy world**: 에셋 추출 없이 CARLA API 만으로 도로/차선(waypoint), 건물·식생 등(박스), 차량(바운딩 박스 + 차량 색 업리프팅), 신호등(상태별 LED 스펙트럼)을 만듭니다. 정밀 형상이 필요하면 `town`/`blueprints` 에 glTF 를 지정하세요 (`docs/carla_live.md`, `docs/carla_export.md`).
segmentation ID 는 CARLA `CityObjectLabel` 과 동일 (차량 14, 도로 1, 건물 3 …).

성능 조절 순서: `render_resolution` → `render.bands.step` → `spp_live` → CARLA `--quality Low`/`--offscreen` → `traffic`.

---

## 8. 라이브러리로 쓰기

C++ (`find_package(spectral)` → `spectral::spectral`):
```cpp
#include <spectral/api.h>
spectral::Renderer r;
r.load_scene_file("scene.json");
spectral::SpectralImage img = r.render();     // img.radiance[(y*W + x)*bands + b]
spectral::Renderer::write_outputs(img, r.scene().output);
```
설치: `cmake --install build --prefix /opt/spectral` 후 `-DCMAKE_PREFIX_PATH=/opt/spectral`.

Python (`source env.sh`):
```python
import numpy as np
from spectral_renderer import Session
s = Session("examples/windshield/scene.json", backend="cuda")
M = np.eye(4)[:3]; M[:, 3] = [0, 0.5, -5]
h = s.spawn({"primitive": "box", "size": [1, 1, 1], "material": {"base_color": [0.6, 0.05, 0.05]},
             "transform": M, "seg_id": 14})
s.set_transform(h, M); s.reset(); s.render(16)
rgb8 = s.preview("srgb")           # (H, W, 3) uint8, GPU 에서 계산
cube = s.image()["radiance"]       # (H, W, B) float32
s.save(npz="out.npz", exr="out.exr", preview_png="out.png")
```

---

## 9. 문제 해결

| 증상 | 확인 / 해결 |
|---|---|
| `setup.sh` 가 `Python headers missing` | `sudo apt install python3.10-dev python3.10-venv` |
| `-- CUDA backend: OFF` 인데 GPU 가 있음 | `nvcc` 가 PATH 에 없음 → `export PATH=/usr/local/cuda/bin:$PATH`, 또는 `scripts/setup.sh --cuda on` |
| `optix.h not found` / OptiX 다운로드 실패 | 네트워크 확인, 또는 `scripts/setup.sh -- -DOPTIX_ROOT=/path/OptiX-SDK` |
| `optixInit failed (OptixResult ...)` | 드라이버가 헤더 버전보다 오래됨 → 드라이버 업데이트, 또는 `-- -DSPECTRAL_OPTIX_TAG=v8.1.0` |
| `CUDA driver version is insufficient` | 드라이버 CUDA 버전 < 툴킷 버전 → 드라이버 업데이트 또는 낮은 툴킷 |
| `--backend cuda` 결과가 CPU 와 다름 | `python scripts/compare_backends.py ...` 출력 첨부해 보고 |
| `ImportError: _spectral` | 다른 Python 으로 빌드됨 → `rm -rf build && scripts/setup.sh` |
| `carla` import 실패 / 버전 경고 | 서버와 client 버전 일치: `pip install carla==0.9.15` (Python 3.10) |
| CARLA 가 바로 종료 | `carla_server.log` 확인, `vulkaninfo`, `sudo apt install libvulkan1 libomp5` |
| 뷰어 오른쪽 패널이 느림 | `render_resolution` 축소, `bands.step` 10→20, `spp_live` 2, `--quality Low` |
| proxy world 가 비어 있음 | 맵 로딩 전 → `carla_town` 확인, `timeout` 증가 |
| 첫 렌더가 몇 초 멈춤 | RGB→스펙트럼 LUT 최초 생성(~8 s, 이후 캐시), Prague 테이블 최초 계산 |

---

## 10. 현재 상태

| 구성 요소 | 상태 |
|---|---|
| CPU 렌더러, glTF/장면 로딩, EXR/NPZ | 구현, C++ 55 test cases 통과 |
| Live Session (증분 업데이트, 프리뷰) + Python 모듈 | 구현, CPU 에서 검증 (이동/삭제 결과가 새로 만든 장면과 bit-identical) |
| CUDA/OptiX 백엔드 | 구현, nvcc 13.x + OptiX 9.0 헤더로 **빌드·링크 검증**, **GPU 실행은 서버에서 첫 검증 필요** |
| Prague sky | 통합·컴파일, 데이터셋으로 실행은 미검증 (gated test 제공) |
| 광학 단계 | 구현, Python 19 tests (Python 3.10 / 3.11) |
| Blender 애드온 | 구현, bpy 4.2 헤드리스 테스트 |
| CARLA 오프라인 변환 / 온라인 뷰어 | 구현, 가짜 CARLA API 로 테스트, **실제 CARLA 서버는 미검증** |
| 재질 스펙트럼 | `approx_*` 는 자리표시자 → 실측 데이터로 교체 필요 |

한계: 780 nm 이상 RGB 텍스처 재질의 NIR 은 가정값(측정 스펙트럼 권장), 안개/비(참여 매질) 미구현, 거친 유리 투과 미구현, 핀홀 카메라(렌즈 효과는 광학 단계).

## 11. 문서
| 문서 | 내용 |
|---|---|
| `docs/architecture.md` | 설계, dense-band 전송, 코드 구조, live 업데이트, 테스트 |
| `docs/gpu_build.md` | CUDA/OptiX 빌드와 검증 |
| `docs/scene_json_schema.md` | 장면 JSON 레퍼런스 |
| `docs/carla_live.md` | CARLA 온라인 뷰어 상세 |
| `docs/carla_export.md`, `docs/blender_export.md` | 장면 소스 |
| `docs/optics_stage.md`, `docs/windshield.md` | 광학 단계, 윈드실드 모델 |
| `docs/sky_plugin.md` | Prague sky, 플러그인 ABI |
| `data/spectra/materials/README.md` | 재질 스펙트럼 출처 |

스크립트: `scripts/setup.sh`, `scripts/install_carla.sh`, `scripts/run_carla_live.sh`, `scripts/compare_backends.py`,
`scripts/check_cuda_build.sh` (GPU 없이 nvcc 빌드 검증), `scripts/check_cuda_compile.sh`.
