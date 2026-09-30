# Architecture

```
 Blender / CARLA ──► glTF assets + scene JSON ──► libspectral (C++) ──► spectral cube (EXR/NPZ)
                                                   │  CPU (OpenMP, BVH)          + depth + seg_id
                                                   └  CUDA/OptiX (IAS/GAS)
 lens vendor data / .zmx / windshield ──► spectral_optics (Python) ──► optics package ──┐
 spectral cube ─────────────────────────────────────────────────────► apply ──► optical image ──► camera pipeline
```

## Dense-band spectral transport (no hero wavelength)
Every path carries a `Spectrum` of all output bands (default 125 bands, 380–1000 nm, 5 nm).
Sampling decisions (light selection, BSDF lobes, directions, Russian roulette on the max band)
are wavelength independent, so each path contributes to every band and there is no per-wavelength
MIS. Hero-wavelength sampling is only needed when directions depend on λ (dispersion); here
refraction uses the reference (d-line) IOR while Fresnel is per band, and chromatic effects of
optics (lens, windshield wedge) are handled by the optics stage. Cost: 2 × 516 B of path state.

## Code layout
| Path | Role |
|---|---|
| `include/spectral/kernel/` | Host/device kernels (no STL/exceptions/virtuals; compiled by g++ and nvcc): math, RNG, sampling, BSDFs, lights, uplifting, textures, camera, integrator |
| `include/spectral/spectrum/` | `Spectrum` (device), `DenseSpectrum` (host, 1 nm), CIE tables, colorimetry, glass, uplift LUT |
| `src/scene/` | `Scene` container (flat arrays → `SceneView`), JSON loader, glTF loader, primitives |
| `src/backend/cpu/` | Two-level binned-SAH BVH, watertight triangle test, tiled OpenMP renderer |
| `src/backend/cuda/` | OptiX pipeline; raygen runs `integrator.h`, hit/miss programs only report hits |
| `src/sky/` | Simple / Prague / plugin sky models, tabulation into an environment table |
| `src/io/` | Multispectral EXR (tinyexr), NPZ (own NPY+ZIP/ZIP64 writer), CSV spectra, preview PNG |
| `src/api/` | `Renderer` (one-shot) and `Session` (persistent scene: spawn/move/hide/recycle instances, dynamic lights, sun, accumulation, preview) |
| `python/` | pybind11 module `spectral_renderer` wrapping `Session` (numpy in/out, GIL released while rendering) |
| `apps/` | `spectral_render` CLI, `spectral_uplift_lut` |
| `optics/` | Python optics stage |
| `tools/` | Blender add-on, CARLA converter (offline), CARLA live viewer (`tools/carla_live`, see `docs/carla_live.md`), data generators |

## Integrator
Unidirectional path tracing with next-event estimation and power-heuristic MIS for area,
environment and sun lights; delta lights (point, spot, directional) via NEE only. Lights are
selected with an alias table weighted by an irradiance estimate at a reference distance.
Glass (dielectric / thin dielectric) is sampled with delta events; with transparent shadows
enabled, shadow rays cross glass with Fresnel + absorption attenuation and transmission events
keep the MIS state of the last diffuse vertex, so light through a windshield is sampled
efficiently (exact for panes; approximate for strongly curved glass). Glass events have their own
budget (`max_glass_events`) and do not consume `max_depth`.

Materials: glTF metallic-roughness (Lambert + GGX with VNDF sampling, Schlick Fresnel,
KHR_materials_specular), smooth dielectric (per-band Fresnel from Sellmeier IOR, Beer absorption
inside closed meshes), thin dielectric (incoherent slab R/T, path-length-corrected internal
transmittance). RGB inputs are uplifted with Jakob–Hanika (2019); measured spectra override them.

## Determinism
Each (seed, pixel, sample) has its own PCG32 stream: images are bit-identical across CPU thread
counts; CPU and GPU agree statistically (floating-point contraction differs).

## Live updates
`Session` keeps the `Scene` and backend alive between frames. Removed instances are hidden (their
index stays valid) and recycled by the next spawn with the same asset/primitive signature; identical
primitives share one mesh. `Backend::update` builds acceleration structures only for new meshes and
rebuilds the top level (CPU TLAS / OptiX IAS, hidden instances masked out), re-uploads the environment
only when it changed, and reuses the film. Preview images (sRGB with auto exposure, single band, false
colour, depth, segmentation) are reduced from the band cube on the device, so a live frame copies
only W·H·3 bytes to the host. On the CPU, an incrementally updated scene renders bit-identically to
the same scene built from scratch.

## Tests
`ctest --test-dir build` (C++, doctest, ~5 s after the one-time uplift table) and
`python -m pytest optics/tests tools/carla_export` (Python); with `-DSPECTRAL_BUILD_PYTHON=ON`,
`PYTHONPATH=build/python python -m pytest python/tests tools/carla_live/tests` (session bindings,
CARLA live viewer against a fake CARLA API). Highlights:
white/emissive furnaces, exact Lambert plane per band, inverse-square law, spherical area light
(NEE+MIS), sun-disk irradiance, importance-sampled HDR environment, ColorChecker under D65
(mean ΔE76 0.04 measured spectra / 0.23 uplifted sRGB), windshield slab/pane transmission and
reflection formulas, transparent shadows, BVH vs brute force, EXR/NPZ round trips (incl. ZIP64),
glTF hierarchy/materials/textures, sky plugin ABI, optics PSF physics, Blender and CARLA exporters.
