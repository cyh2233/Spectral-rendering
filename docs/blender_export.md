# Blender → spectral renderer

Add-on: `tools/blender_addon/spectral_exporter` (Blender 3.6+ / 4.x). Install by zipping the
folder (or symlinking it into Blender's `scripts/addons`) and enabling "Spectral Renderer Exporter".

- **File › Export › Spectral Scene (.json)** writes `<name>.glb` + `<name>.json`.
- **Properties › Render › Spectral**: samples, bounces, band grid, backend, sky model
  (simple / Prague + dataset / plugin), default material spectra, and **Render Spectral**
  (runs `spectral_render`, set its path in the add-on preferences, and loads the sRGB preview).

## What is exported
| Blender | Scene JSON |
|---|---|
| Visible mesh objects | one glTF asset (Y-up), node names = object names |
| Object property `seg_id` (int) | `seg_id_by_node` (`^name$`) |
| Principled BSDF (base colour, metallic, roughness, normal map, emission, alpha, transmission) | glTF PBR (+KHR extensions) → uplifted spectra; transmission → thin dielectric |
| Material properties `spectral_*` | `materials.overrides` for that material (see below) |
| Active perspective camera | pinhole `matrix`, `fov_deg`, `fov_axis` (sensor fit), resolution × percentage |
| Sun lamp | sky `sun.direction`; simple sky: `sun.irradiance` = strength over 380–780 nm |
| Point / spot lamps | radiant intensity = power / 4π over 380–780 nm; spot cone and blend |
| Area lamps | not exported (use emissive meshes) |

Material custom properties (Object › Material › Custom Properties):
`spectral_reflectance` (library name, number or JSON), `spectral_bsdf`
(`pbr`/`dielectric`/`thin_dielectric`), `spectral_glass` (e.g. `soda_lime`), `spectral_ior`,
`spectral_transmittance`, `spectral_internal_transmittance` + `spectral_thickness_mm`,
`spectral_emission`, `spectral_emission_scale`, `spectral_roughness`, `spectral_metallic`,
`spectral_specular`. Light properties: `spectral_emission` (library name, e.g. `cie_hp1`) or
`spectral_temperature` (K, blackbody; default 6500).

Units: Blender lamp values are *not* physical. The exporter writes them as radiometric values
over 380–780 nm (sun W/m², point/spot W/sr); set physically meaningful values (clear-day sun
≈ 400–800 W/m² in the visible) or use the Prague sky. Default simple-sky radiance: 0.0005 × D65
(≈ 0.05 W/(m² sr nm) at 560 nm).

Test (headless, with `pip install bpy==4.2.0`):
`python tools/blender_addon/test_export_blender.py build/spectral_render /tmp/out`
builds a road scene with a windshield, renders it, and checks segmentation positions and colours.
