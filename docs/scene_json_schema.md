# Scene JSON reference

A scene file describes one frame. Relative paths are resolved against the scene file's directory.
`${SPECTRAL_DATA}` expands to the data directory (`$SPECTRAL_DATA_DIR`, else the source/install
`data/`), `${SCENE_DIR}` to the scene directory, `${NAME}` to environment variables.
Comments (`//`, `/* */`) are allowed.

World frame: right-handed, **+Y up**, metres (glTF convention). Cameras look down their local −Z.

## Top level

| Key | Type | Description |
|---|---|---|
| `version` | int | Informational (2). |
| `camera` | object | Pinhole camera, see below. |
| `assets` | object | `name -> {"path": "file.glb"}` glTF 2.0 assets (.gltf / .glb). |
| `instances` | array | Asset or primitive instances. |
| `materials` | object | Spectrum library, uplifting, named materials, overrides. |
| `lights` | array | Point / spot / directional / sky / envmap. Area lights = emissive materials. |
| `render` | object | Sampling, bands, backend. |
| `output` | object | EXR / NPZ / preview paths and metadata. |

## camera
| Key | Default | |
|---|---|---|
| `type` | `"pinhole"` | Only pinhole: lens effects are applied by the optics stage. |
| `resolution` | `[64, 64]` | `[width, height]` in pixels. |
| `fov_deg` + `fov_axis` | `60`, `"horizontal"` | `horizontal`, `vertical` or `diagonal`. |
| `focal_length_mm` + `sensor_width_mm` [+ `sensor_height_mm`] | | Alternative to `fov_deg`. |
| `matrix` | | 3x4 or 4x4 row-major camera-to-world matrix. |
| `position`, `look_at`, `up` | `[0,0,0]`, `-Z`, `[0,1,0]` | Look-at pose. |
| `position`, `rotation` | | Quaternion `[x, y, z, w]`. |

## instances
Asset instance:
```json
{ "asset": "town", "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "seg_id": 0,
  "seg_id_by_material": { "(?i)road": 1 }, "seg_id_by_node": { "^Car_01$": 14 },
  "material_overrides": [ { "match": "(?i)paint", "reflectance": "approx_car_paint_red" } ] }
```
Primitive instance: `"primitive": "sphere" | "quad" | "box" | "disk"` with `radius`, `size`,
`segments`, `rings`, `flip_normals`, and `"material"` (a name from `materials.definitions` or an
inline material object). Quads/disks lie in the XZ plane facing +Y.

Transforms: `transform` (4x4/3x4 row-major), or `translate` + `rotate_deg` (extrinsic X, then Y,
then Z) or `rotation` (quaternion) + `scale` (number or `[x, y, z]`), composed as T·R·S.

`seg_id` (uint32) is written to the segmentation AOV. Patterns are ECMAScript regular expressions;
a leading `(?i)` makes them case-insensitive. Later rules win.

## Spectra
Wherever a spectrum is expected (`reflectance`, `transmittance`, `emission`, `absorption`, `ior`,
light `emission`, sky radiances):

| Form | Meaning |
|---|---|
| `0.5` | Constant. |
| `"approx_concrete"` | Library CSV by file stem (all `*.csv` under `data/spectra` and `materials.library_dirs`). |
| `"colorchecker_babel:red"` | Column of a multi-column library CSV. |
| `{"type": "constant", "value": v}` | |
| `{"type": "csv", "path": "...", "column": "name" \| index}` | |
| `{"type": "samples", "wavelengths": [...], "values": [...], "extrapolation": "clamp"\|"zero"}` | |
| `{"type": "blackbody", "temperature": K}` | Planck radiance, W/(m² sr nm). |
| `{"type": "cie", "name": "D65" \| "A" \| "E"}` | D65/A relative (100 at 560 nm). Other CIE sources: library `cie_*`. |
| `{"type": "rgb", "value": [r, g, b]}` | Linear sRGB, uplifted (reflectance or illuminant style by context). |
| `{"type": "glass", "name": "soda_lime"}` | Refractive index from `data/glass/catalog.json`. |

Modifiers (any object form): `"scale"`, `"normalize": "peak"`,
`"photometric": X` (so that 683·∫S·ȳ = X: cd for intensities, cd/m² for radiances, lx for irradiances),
`"integral": X` with optional `"integral_range_nm": [380, 780]` (∫S dλ = X over the range).

Every spectrum is box-averaged onto the output band grid at load time.

## materials
```json
"materials": {
  "library_dirs": ["./my_spectra"],
  "uplift": { "hold_lambda": 780, "resolution": 64, "lut": "optional/path.coeff" },
  "use_default_overrides": false,
  "definitions": { "name": { ...material fields... } },
  "overrides": [ { "match": "(?i)asphalt", "asset": "optional asset-key regex", ...material fields... } ]
}
```
Material fields (applied on top of glTF materials):

| Field | Description |
|---|---|
| `bsdf` | `pbr` (default), `dielectric` (solid glass; closed geometry), `thin_dielectric` (single-surface pane). |
| `base_color` | Linear RGB(A) factor (uplifted unless `reflectance` is given). |
| `reflectance` | Spectral reflectance; replaces base colour/texture (albedo for diffuse, F0 for metals). |
| `metallic`, `roughness`, `specular` | glTF metallic-roughness; `specular` = KHR_materials_specular (0 = pure Lambert). |
| `ior` | Number (reference IOR) or spectrum (per-band Fresnel). |
| `glass` | Catalog name: sets per-band IOR and the d-line reference IOR. |
| `transmittance` | Thin dielectric: internal transmittance per pass at normal incidence. |
| `absorption` | Solid dielectric: absorption coefficient [1/m]. |
| `internal_transmittance` + `thickness_mm` | Converted to `absorption = -ln(T)/d`. |
| `emission`, `emission_scale`, `emissive` | Spectral radiance (W/m²/sr/nm) or RGB emissive factor. Emissive meshes become area lights. |
| `alpha_mode` (`opaque`\|`mask`\|`blend`), `alpha_cutoff`, `double_sided` | |

Uplifting: Jakob & Hanika (2019) sigmoid-polynomial table for sRGB, generated on first use into
the cache directory (`$SPECTRAL_CACHE_DIR` or `~/.cache/spectral`, ~8 s). Uplifted reflectances
are held constant beyond `hold_lambda` (NIR is not recoverable from RGB; assign measured spectra
to NIR-relevant materials).

## lights
| Type | Keys |
|---|---|
| `point` | `position`, `emission` (radiant intensity, W/(sr nm)), `scale` |
| `spot` | + `direction`, `cone_inner_deg`, `cone_outer_deg` (half angles, smoothstep fall-off) |
| `directional` | `direction` (of travel), `emission` (irradiance normal to the light, W/(m² nm)) |
| `sky` `simple` | `sky_radiance` (constant), `sun`: `{elevation_deg, azimuth_deg}` or `{direction}`, plus `radiance` or `irradiance`, `half_angle_deg` (0.2667) |
| `sky` `prague` | `dataset` (Prague Sky Model .dat; SWIR dataset recommended for >760 nm), `sun`, `visibility_km`, `ground_albedo`, `altitude_m`, `table_resolution` ([512, 256]), `cache` |
| `sky` `plugin` | `library` (shared library implementing `spectral/sky/sky_plugin.h`), `options` (passed as JSON) |
| `envmap` | `path` (EXR/HDR/PNG, linear RGB), `resolution` ([512, 256]), `scale`, `rotation_deg` |

Sun angles: elevation above the horizon; azimuth from −Z ("north") towards +X ("east").
All lights accept `"enabled": false`.

## render
| Key | Default | |
|---|---|---|
| `spp` | 64 | Samples per pixel. |
| `spp_per_pass` | 4 | Progress granularity. |
| `max_depth` | 8 | Non-glass scattering events. |
| `max_glass_events` | 16 | Glass interactions (do not consume `max_depth`). |
| `rr_depth` | 3 | Russian roulette start. |
| `bands` | `{min: 380, max: 1000, step: 5}` | Output band grid (≤ 128 bands by default). |
| `seed` | 0 | Deterministic per (seed, pixel, sample). |
| `backend` | `auto` | `cpu`, `cuda`, `auto`. |
| `threads` | 0 | CPU threads (0 = all). |
| `transparent_shadows` | true | Shadow rays pass glass with Fresnel/absorption attenuation. |
| `depth_mode` | `distance` | Depth AOV: `distance` along the ray or camera `z`. |
| `clamp` | 0 | >0 clamps per-sample radiance (biased; off by default). |
| `light_reference_distance` | 10 | Light-selection heuristic scale (m). |

## output
`exr`, `npz`, `preview_png` (paths), `exr_half` (bool), `metadata` (object, copied into the files
together with camera intrinsics and band information).

EXR channels: `L_0380` … `L_1000` (float or half), `depth` (float), `seg_id` (uint).
NPZ arrays: `radiance` (H, W, B) float32, `wavelengths` (B), `depth` (H, W), `seg_id` (H, W) uint32,
`metadata` (JSON bytes), `spp`.
Units: spectral radiance W/(m² sr nm), band-averaged. Sky pixels: depth = FLT_MAX, seg_id = 0xFFFFFFFF.
AOVs come from a straight probe ray through the pixel centre that skips glass (a windshield does
not hide the scene in the depth/segmentation maps).
