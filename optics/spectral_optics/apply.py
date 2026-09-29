"""Applies an OpticsPackage to a rendered spectral radiance cube (pinhole) -> optical image.

Steps (per band): lens distortion warp, shift-variant PSF convolution (bilinear blending of the
per-field convolutions over a rectangular field grid), relative illumination, lens transmission.
Optionally converts scene radiance to sensor irradiance, E = pi T L / (4 N^2), the standard
camera equation for distant objects (relative illumination supplies the off-axis fall-off; if the
package has none, cos^4 is used).
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from .package import OpticsPackage
from .pupil import resample_psf


def camera_from_metadata(meta_json: str):
    meta = json.loads(meta_json) if meta_json else {}
    cam = meta.get("camera")
    if not cam:
        raise ValueError("render metadata has no camera intrinsics (re-render with a current libspectral)")
    return float(cam["tan_half_fov_x"]), float(cam["tan_half_fov_y"])


def _field_grid(pkg: OpticsPackage):
    xs = np.unique(np.round(pkg.fields_deg[:, 0], 9))
    ys = np.unique(np.round(pkg.fields_deg[:, 1], 9))
    if len(xs) * len(ys) != len(pkg.fields_deg):
        raise ValueError("package fields must form a rectangular grid (use vendor.expand_rotational)")
    index = {}
    for i, (x, y) in enumerate(np.round(pkg.fields_deg, 9)):
        index[(x, y)] = i
    return xs, ys, index


def _bilinear_weights(coords, axis_vals):
    """For each coordinate, returns (i0, i1, t) on the sorted axis (clamped)."""
    if len(axis_vals) == 1:
        z = np.zeros_like(coords, dtype=int)
        return z, z, np.zeros_like(coords)
    i0 = np.clip(np.searchsorted(axis_vals, coords) - 1, 0, len(axis_vals) - 2)
    t = np.clip((coords - axis_vals[i0]) / (axis_vals[i0 + 1] - axis_vals[i0]), 0.0, 1.0)
    return i0, i0 + 1, t


def apply_optics(radiance: np.ndarray, wavelengths_nm: np.ndarray, tan_half_fov_x: float, tan_half_fov_y: float,
                 pkg: OpticsPackage, irradiance: bool = False, apply_distortion: bool = True,
                 psf_support_um: Optional[float] = None) -> np.ndarray:
    """radiance: (H, W, B) pinhole render. Returns an (H, W, B) optical image on the same pixel grid
    (sensor pixel pitch = 2 f tan(hfov_x) / W)."""
    H, W, B = radiance.shape
    f = pkg.focal_length_mm
    pitch_mm = 2.0 * f * tan_half_fov_x / W
    pitch_y_mm = 2.0 * f * tan_half_fov_y / H
    if abs(pitch_mm - pitch_y_mm) > 1e-6 * pitch_mm:
        raise ValueError("non-square pixels are not supported")
    # Sensor coordinates (mm) of pixel centres, +x right, +y up.
    u = (np.arange(W) + 0.5 - W / 2) * pitch_mm
    v = (H / 2 - (np.arange(H) + 0.5)) * pitch_mm
    U, V = np.meshgrid(u, v)
    h_real = np.hypot(U, V)
    theta = pkg.field_from_height_mm(h_real) if apply_distortion else np.degrees(np.arctan(h_real / f))
    out = np.empty_like(radiance, dtype=np.float64)

    # 1) Distortion: sample the pinhole image at the ideal position of each real sensor point.
    if apply_distortion and pkg.distortion_field_deg is not None:
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(h_real > 0, f * np.tan(np.radians(theta)) / h_real, 1.0)
        src_u, src_v = U * scale, V * scale
        col = src_u / pitch_mm + W / 2 - 0.5
        row = H / 2 - 0.5 - src_v / pitch_mm
        for b in range(B):
            out[..., b] = ndimage.map_coordinates(radiance[..., b], [row, col], order=1, mode="nearest")
    else:
        out[:] = radiance

    # 2) Shift-variant PSF.
    field_x = np.degrees(np.arctan2(U, f)) if not apply_distortion else np.where(
        h_real > 0, theta * U / np.maximum(h_real, 1e-12), 0.0)
    field_y = np.degrees(np.arctan2(V, f)) if not apply_distortion else np.where(
        h_real > 0, theta * V / np.maximum(h_real, 1e-12), 0.0)
    xs, ys, index = _field_grid(pkg)
    ix0, ix1, tx = _bilinear_weights(field_x, xs)
    iy0, iy1, ty = _bilinear_weights(field_y, ys)
    K = pkg.psf.shape[-1]
    extent_um = psf_support_um or K * pkg.psf_pixel_um
    size = int(np.ceil(extent_um / (pitch_mm * 1e3))) | 1
    size = max(size, 3)
    weights = {}
    for i in range(len(xs)):
        for j in range(len(ys)):
            w = np.zeros((H, W))
            for a, wa in ((ix0, 1 - tx), (ix1, tx)):
                for b_, wb in ((iy0, 1 - ty), (iy1, ty)):
                    w += np.where((a == i) & (b_ == j), wa * wb, 0.0)
            if np.any(w > 0):
                weights[(i, j)] = w
    blurred = np.zeros_like(out)
    for b in range(B):
        img = out[..., b]
        acc = np.zeros((H, W))
        for (i, j), w in weights.items():
            fi = index[(np.round(xs[i], 9), np.round(ys[j], 9))]
            k = resample_psf(pkg.psf_at(float(wavelengths_nm[b]), fi), pkg.psf_pixel_um, pitch_mm * 1e3, size)
            acc += w * fftconvolve(img, k, mode="same")
        blurred[..., b] = acc
    out = blurred

    # 3) Relative illumination / 4) transmission / 5) radiance -> irradiance.
    if pkg.ri is not None:
        out *= pkg.relative_illumination(theta)[..., None]
    elif irradiance:
        out *= (np.cos(np.radians(theta)) ** 4)[..., None]
    if pkg.transmission is not None:
        out *= np.interp(wavelengths_nm, pkg.transmission_nm, pkg.transmission)[None, None, :]
    if irradiance:
        out *= np.pi / (4.0 * pkg.f_number ** 2)
    return out.astype(np.float32)


def apply_to_npz(in_path: str, pkg: OpticsPackage, out_path: str, irradiance: bool = False) -> None:
    """Reads a renderer NPZ, applies the optics and writes an NPZ with the same keys."""
    with np.load(in_path) as z:
        data = {k: z[k] for k in z.files}
    meta = bytes(data["metadata"]).decode() if "metadata" in data else ""
    tx, ty = camera_from_metadata(meta)
    data["radiance"] = apply_optics(data["radiance"], data["wavelengths"], tx, ty, pkg, irradiance=irradiance)
    m = json.loads(meta) if meta else {}
    m["optics"] = {"applied": True, "irradiance": irradiance, "package": pkg.metadata,
                   "units": "W/(m^2 nm)" if irradiance else "W/(m^2 sr nm) (blurred radiance)"}
    data["metadata"] = np.bytes_(json.dumps(m))
    np.savez(out_path, **data)
