# CARLA live viewer (online)

`tools/carla_live/spectral_live.py` connects to a running CARLA server. It drives an ego vehicle with a camera and
mirrors the world into a persistent spectral render session. A window shows the CARLA RGB camera and the spectral
render (sRGB preview or other views) side by side.

> 요약: 서버에 CARLA 0.9.15/0.9.16 설치 → `-DSPECTRAL_ENABLE_CUDA=ON -DSPECTRAL_BUILD_PYTHON=ON` 로 빌드 →
> `./CarlaUE4.sh` 실행 → `PYTHONPATH=build/python python tools/carla_live/spectral_live.py --config my.json`.
> 에셋(glTF) 추출이 안 되면 CARLA Python API만으로 만든 **proxy world** (도로/차선/건물 박스/차량 박스/신호등)가
> 자동으로 사용됩니다. `M` 키로 live ↔ quality 모드 전환.

```
CARLA server (UE4, GPU) ──TCP 2000──► spectral_live.py
                                       ├─ CarlaRig      synchronous mode, ego + autopilot, RGB camera, traffic
                                       ├─ CarlaSync     actors/lights/sun/camera → Session updates each tick
                                       │   └─ proxy_world  roads/markings from waypoints, env-object boxes
                                       ├─ spectral_renderer.Session   (C++ via pybind11, CUDA/OptiX or CPU)
                                       │   incremental BLAS/TLAS updates, device-side preview (only RGB8 copied)
                                       └─ Viewer (pygame)  CARLA RGB | spectral preview + HUD
```

## 1. Install CARLA (Ubuntu 22.04)

1. Download the packaged release from GitHub (`CARLA_0.9.15.tar.gz` or `CARLA_0.9.16.tar.gz`). For the other
   towns, also download `AdditionalMaps_0.9.1x.tar.gz` and extract it into the same directory.
2. Check that Vulkan works (`vulkaninfo | head`). UE4 on Blackwell needs a recent NVIDIA driver (570+). The same
   driver serves CUDA 12.8+ for the renderer.
3. Start the server:
   ```bash
   ./CarlaUE4.sh -quality-level=Epic            # with the spectator window
   ./CarlaUE4.sh -RenderOffScreen -quality-level=Low   # no UE window, cheaper; the viewer still gets RGB
   ```

## 2. Python environment

Use Python 3.10. Both 0.9.15 and 0.9.16 publish wheels for it.
```bash
python3.10 -m venv ~/venv-carla && source ~/venv-carla/bin/activate
pip install carla==0.9.15        # must match the server version (or the .whl in PythonAPI/carla/dist)
pip install numpy pygame
```

## 3. Build the renderer with the Python module

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSPECTRAL_ENABLE_CUDA=ON -DOPTIX_ROOT=/path/to/OptiX-SDK -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DSPECTRAL_BUILD_PYTHON=ON -DPython3_EXECUTABLE=$(which python)
cmake --build build
ctest --test-dir build
PYTHONPATH=build/python python -m pytest python/tests tools/carla_live/tests   # fake CARLA, no server needed
```
The module is written to `build/python/spectral_renderer/`. It must be built with the same Python as the venv.
See `docs/gpu_build.md` for the GPU validation steps (`compare_backends.py`) to run first.

## 4. Run

```bash
cp tools/carla_live/config.example.json my_live.json     # edit host, town, resolution, ...
PYTHONPATH=build/python python tools/carla_live/spectral_live.py --config my_live.json
# options: --host 10.0.0.5 --port 2000 --backend cuda|cpu --frames N --headless
```
The script switches the world to synchronous mode with `fixed_delta_seconds`. On exit it restores the previous
settings and destroys its actors. It also loads `carla_town` if that town differs from the current map.

| Key | Action |
|---|---|
| `M` | Live ↔ quality mode |
| `V` | View: sRGB → single band → NIR false colour (`false_color_nm`) → depth → segmentation |
| `[` `]` | Band for the single-band view |
| `+` `-` | Exposure (EV) on top of auto exposure |
| `P` | Pause (CARLA stops ticking; in live mode the render keeps refining) |
| `S` | Save the current cube (`save_dir/snapshot_N.npz`, `.exr`, `_srgb.png`) |
| `R` | Record: in quality mode, every finished frame is saved (`frame_N.*`) |
| `Esc` | Quit |

**Live mode** resets accumulation on every CARLA tick and renders `spp_live` samples per pixel at
`render_resolution`. The world keeps moving, so the preview is noisy but interactive.

**Quality mode** holds CARLA; in synchronous mode the world does not advance. It renders `spp_chunk` samples at a
time up to `spp_quality`, refreshing the window between chunks. Then it ticks once, so each saved cube is converged
and matches exactly one CARLA frame. The CARLA RGB image shown next to it comes from the same tick.

## 5. Geometry sources

For each asset, the viewer uses a glTF if you configure one and falls back to a proxy otherwise.

| Source | Needs | Fidelity |
|---|---|---|
| `town.<MapName>.path` glTF (UE editor → FBX/glTF, or Blender, see `docs/carla_export.md`) | Asset export | Full mesh + PBR textures |
| Proxy world (default) | Only the packaged CARLA + Python API | Roads and lane markings from `map.generate_waypoints` (lane widths, solid/broken, white/yellow), a ground plane, oriented boxes from `world.get_environment_objects()` with per-class materials (building, vegetation, pole, fence, …) |
| `blueprints.<type_id>.path` glTF | Vehicle export | Mesh in place of the bounding box |
| Proxy actors (default) | — | Vehicle bounding boxes painted with the actor's `color` attribute (RGB uplift), walker boxes |

Some objects are always generated, including when a town glTF is used:
- **Traffic lights** come from `get_light_boxes()`, with emissive LED spectra for the current state.
- **Headlights** come from `get_light_state()` (low or high beam) as spectral spot lights.

Segmentation IDs are CARLA's `CityObjectLabel` values, so they can be compared with CARLA's semantic camera. The
sky is `0xFFFFFFFF`.

The proxy ego vehicle is not rendered, because the camera sits inside its box. With a glTF ego blueprint, set
`"render_ego": true` to see the hood, cabin and windshield.

The sun direction follows `world.get_weather()` (`sun_altitude_angle` / `sun_azimuth_angle`). For the Prague sky,
set `"sky": {"model": "prague", "dataset": ...}`. Its tables are cached per quantised sun angle, so a
time-of-day change costs one re-tabulation.

Proxy materials use approximate spectra. Override them with measured data through the `materials` key, which uses
the same format as the scene JSON `materials.definitions`.

## 6. Performance on one GPU (CARLA + OptiX)

CARLA and the renderer share the GPU. Memory is not an issue at 96 GB: CARLA uses about 6–10 GB, and the renderer
uses mostly scene size plus `W·H·bands·4` bytes for the film. The main knobs, in order of effect:

1. `render_resolution` (e.g. 960×540 for live, full resolution for quality).
2. `render.bands.step`: 10 nm (63 bands) for live, 5 nm (125 bands) for final cubes. Cost is roughly linear
   in the number of bands for shading and film.
3. `spp_live` and `render.max_depth`.
4. CARLA's `-quality-level=Low` and `-RenderOffScreen`, which free GPU time for the renderer.
5. `traffic`: the number of actors. Updates are incremental (moving an actor rebuilds only the TLAS), so this
   matters mostly for CARLA itself.

The HUD shows fps, accumulated spp, the object count and the backend. Only the 8-bit preview crosses PCIe per frame;
the full cube is copied only when saving.

## 7. Troubleshooting

- `ImportError: _spectral`: the module was built for a different Python. Re-run CMake with
  `-DPython3_EXECUTABLE` pointing to the venv Python.
- `RuntimeError: time-out` from CARLA: the server is still loading the map. Increase `timeout` in the config.
- The window is black on the right: check that `backend` is `cuda` and the terminal shows `(cuda backend)`.
  Try `--backend cpu` at small resolution to separate renderer and GPU issues.
- The world is empty in proxy mode: `generate_waypoints` found no roads. The map may not have loaded; check
  `carla_town`.
- Version mismatch warnings: the `carla` wheel and the server must be the same 0.9.x.
