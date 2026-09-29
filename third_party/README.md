# Vendored third-party code

| Directory | Project | Version | License | Use |
|---|---|---|---|---|
| `nlohmann/` | nlohmann/json | v3.11.3 | MIT | Scene JSON |
| `tinyexr/` | syoyo/tinyexr (+ miniz) | v1.0.9 | BSD-3 / MIT | Multispectral EXR I/O |
| `tinygltf/` | syoyo/tinygltf | v2.9.3 | MIT | glTF 2.0 loading |
| `stb/` | nothings/stb (image, image_write) | from tinygltf v2.9.3 | MIT / Public domain | PNG/JPG/HDR textures, preview PNG |
| `doctest/` | doctest/doctest | v2.4.11 | MIT | Unit tests |
| `prague_sky_model/` | PetrVevoda/pragueskymodel | master (2024) | Apache-2.0 | Spectral sky (dataset downloaded separately) |
| `rgb2spec/` | mitsuba-renderer/rgb2spec | master (721145d) | BSD-3 | Jakob-Hanika RGB->spectrum table optimizer |

## Local patches

- `rgb2spec/rgb2spec_opt.cpp`: added `return 0;` at the end of `main()`. The file is compiled as a
  regular function (`-Dmain=spectral_rgb2spec_main`), where falling off the end is undefined
  behaviour (observed: the optimizer looped and rewrote its output at -O3).

## Prague Sky Model dataset

Not included (hundreds of MB). Download from the links in the upstream README:
- `PragueSkyModelDatasetSWIR.dat` (547 MB, 280-2480 nm, ground level) -- recommended here because the
  renderer's default band grid extends to 1000 nm.
- `PragueSkyModelDatasetGroundInfra.dat` / full dataset (320-760 nm) -- visible range only.
