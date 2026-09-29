# Optics stage (`optics/spectral_optics`)

The renderer outputs a **pinhole** spectral radiance cube. The optics stage converts it into the
optical image a real camera forms: lens (and windshield) PSFs per wavelength and field, lens
distortion, relative illumination, lens transmission and, optionally, the radiance-to-irradiance
camera equation E = π T L / (4 N²).

```bash
pip install -e optics[rayoptics,test]      # numpy, scipy (+ rayoptics for prescriptions)
python -m pytest optics/tests               # 19 tests
```

## Optics package (`*.optics.npz`)
| Array | Shape | |
|---|---|---|
| `wavelengths_nm` | (L,) | PSF wavelengths; linear interpolation in between, clamped outside |
| `fields_deg` | (F, 2) | Field angles (x, y); a rectangular grid for `apply` |
| `psf` | (L, F, K, K) | Sensor-plane PSFs, sum 1, upright image convention (row 0 = top) |
| `psf_pixel_um` | scalar | PSF sample spacing |
| `focal_length_mm`, `f_number` | scalars | |
| `distortion_field_deg`, `distortion_height_mm` | (D,) | Real image height vs field (optional) |
| `ri_field_deg`, `ri` | (R,) | Relative illumination (optional; cos⁴ used for irradiance otherwise) |
| `transmission_nm`, `transmission` | (T,) | Lens spectral transmittance (optional) |

## Sources
1. **Vendor data for a black-box lens** (Zemax `.ZBB` cannot be traced without OpticStudio): ask the
   vendor for OpticStudio exports and describe them in a manifest:
   ```json
   { "focal_length_mm": 6.0, "f_number": 1.8, "psf_pixel_um": 0.5, "psf_size": 64,
     "psf": [ { "file": "psf_550_f00.txt", "wavelength_nm": 550, "field_deg": [0, 0] }, ... ],
     "mtf": [ { "file": "mtf_850_f20.csv", "wavelength_nm": 850, "field_deg": [0, 20],
                "columns": {"freq": 0, "tangential": 1, "sagittal": 2} } ],
     "distortion": { "file": "distortion.txt", "kind": "percent", "columns": {"field_deg": 0, "value": 1} },
     "relative_illumination": { "file": "ri.txt" },
     "transmission": { "file": "transmission.csv" },
     "rotational_symmetry": true, "max_field_deg": 30, "field_grid": 5 }
   ```
   - PSF files: Zemax "Huygens PSF"/"FFT PSF" text listings (UTF-16 or UTF-8; the "Data spacing is
     … µm" and "0.5500 µm at 10.00 (deg)" headers are parsed), `.npy`, or `.csv` matrices.
     Row 0 must be the top of the image; state `pixel_um` if the file has no spacing header.
   - MTF files: frequency (cycles/mm) with tangential/sagittal columns. Converted with a
     zero-phase elliptical OTF — symmetric PSFs only (no coma asymmetry). Prefer PSF exports.
   - Distortion: `percent` = Zemax F-tan(θ) distortion, or `height_mm` = real image height.
   - `rotational_symmetry`: samples along +y are rotated/interpolated to an N×N field grid.
   Build: `spectral-optics vendor manifest.json -o lens.optics.npz`.
2. **Prescriptions** (`.zmx`, CODE V `.seq`) with RayOptics:
   `spectral-optics prescription lens.zmx --fields 0 10 20 30 --wavelengths 450 550 650 850 -o lens.optics.npz`.
   PSFs = FFT of the exit-pupil OPD (orientation validated against RayOptics spot diagrams),
   distortion from real chief rays.
3. **Ideal lens + windshield**: `spectral-optics windshield ...` (see docs/windshield.md).
4. **Combination**: `spectral-optics combine lens.optics.npz ws.optics.npz -o camera.optics.npz`.

## Applying
`spectral-optics apply render.npz camera.optics.npz -o optical.npz [--irradiance]`

The sensor pixel pitch is implied by the render: p = 2 f tan(hfov_x) / W, i.e. render the pinhole
image with the lens's focal length and the sensor's pixel count so pixels map 1:1 onto the
sensor. Steps per band: distortion warp → shift-variant PSF (bilinear blend of per-field
convolutions) → relative illumination → transmission → camera equation. Output keeps the NPZ
layout; metadata gains an `optics` record.

Limitations: PSFs are sampled on a field grid (blend between nodes); object distance is infinity
(no focus breathing); defocus/aberrations of the vendor lens at wavelengths outside the supplied
data are clamped to the nearest wavelength.
