# Windshield simulation

The windshield is split between the renderer and the optics stage so that each effect is
modelled exactly once.

| Effect | Where | How |
|---|---|---|
| Geometric refraction / image displacement | Renderer | Glass geometry (`bsdf: dielectric` slab or `thin_dielectric` pane) |
| Double images / ghosts (reflections at both surfaces) | Renderer | Solid slab: two real surfaces, multiple internal reflections |
| Veiling glare (dashboard / sky reflections) | Renderer | Fresnel reflection of the pane, per band |
| Spectral transmittance, green tint, IR-cut coatings | Renderer | `internal_transmittance` + `thickness_mm`, or `absorption`; thin panes: `transmittance` |
| Sunlight through the glass onto the interior | Renderer | Transparent shadows (Fresnel + absorption on shadow rays) |
| Blur: astigmatism / defocus from a raked curved windshield | Optics stage | `spectral_optics.windshield` wavefront model |
| Lateral colour from wedge (HUD) glass | Optics stage | Chromatic tilt kept, reference tilt removed |

Why: the renderer refracts rays with the reference-wavelength IOR (no angular dispersion; per-band
Fresnel only) and traces a pinhole camera, so it cannot produce aperture-dependent blur. The optics
stage traces full pupils through the windshield but removes the reference-wavelength tilt so the
geometric displacement is not applied twice.

## Renderer setup
```json
"windshield": { "bsdf": "dielectric", "glass": "soda_lime",
                "internal_transmittance": "approx_windshield_green_5mm", "thickness_mm": 5.0 }
```
Model the glass as a closed slab (e.g. a `box` primitive or a solidified mesh from Blender) for
ghosts. A `thin_dielectric` single surface is cheaper (no ghost separation) and uses
`transmittance` as the per-pass internal transmittance at normal incidence (path length scales
with 1/cos θt). Glass events do not consume `max_depth` (budget: `max_glass_events`).
Laminated glass (glass/PVB/glass) can be modelled with three nested slabs; `pvb_approx` exists in
the glass catalog (approximate).

## Optics-stage setup
```bash
spectral-optics windshield --focal-length 6 --f-number 1.8 --fov 60 34 --grid 3 \
    --rake 62 --radius 4000 --distance 60 --wedge-mrad 0.5 -o ws.optics.npz
spectral-optics combine lens.optics.npz ws.optics.npz -o camera.optics.npz
spectral-optics apply render.npz camera.optics.npz -o optical.npz --irradiance
```
`rake` is the angle between the windshield normal and the optical axis at the axis crossing,
`radius` the inner-surface radius of curvature (outer surfaces concentric, `inf` = flat),
`distance` the pupil-to-glass distance, `wedge-mrad` a HUD wedge. By default the on-axis defocus at
550 nm is removed, as when a camera is focused through the windshield (`--no-refocus` keeps it).
`combine` convolves the lens PSFs with the windshield *aberration kernel* (windshield PSF Wiener-
deconvolved by the ideal-lens PSF) so diffraction is not counted twice.

## Validation (tests)
- Thin pane: T = (1−F)²τ / (1 − F²τ²) per band with Sellmeier IOR (renderer, <1 %).
- Solid slab with absorption: (1−F)² e^{−αd} / (1 − F² e^{−2αd}) (renderer, <1 %).
- Pane reflectance (ghost path) R = F + (1−F)²F / (1−F²) (renderer, <3 %).
- Sunlight through a pane with transparent shadows: exact to 1e-4.
- Flat parallel plate in collimated space: aberration-free PSF (optics).
- Wedge: lateral colour magnitude f·Δn·α within 15 % and correct direction (optics).
- Curved raked windshield: astigmatic PSF (optics).
