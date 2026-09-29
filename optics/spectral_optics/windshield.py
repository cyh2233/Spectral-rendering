"""Windshield wavefront / PSF model by exact vector ray tracing.

Geometry (camera frame, millimetres): the entrance pupil is a disk of diameter D at the origin in
the z = 0 plane; the optical axis is +z (towards the scene), +y is up, +x right.
The windshield is a stack of layers (e.g. glass / PVB / glass) bounded by concentric spheres
(or parallel planes when radius = inf). Its inner surface crosses the optical axis at distance
`distance_mm`; its normal there is the axis direction tilted by `rake_deg` about +x (the top of a
windshield leans towards the camera). A `wedge_mrad` rotates the outer surfaces about the same axis
(head-up-display wedge), creating chromatic prism deviation (lateral colour).

For each field direction s (scene direction as seen through the glass) a plane wave is traced from
the scene through the stack to the pupil plane. The optical path to the pupil plane gives the pupil
wavefront; the reference-wavelength piston and tilt are removed (the renderer already models the
geometric refraction), optionally the on-axis defocus (camera focused through the windshield), and
the PSF of an otherwise ideal lens follows by FFT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from scipy.interpolate import griddata

from . import glass
from .package import OpticsPackage
from .pupil import fit_remove, psf_from_wavefront, pupil_grid, resample_psf


@dataclass
class Windshield:
    layers: List[Tuple[str, float]] = field(
        default_factory=lambda: [("soda_lime", 2.1), ("pvb_approx", 0.76), ("soda_lime", 2.1)])
    rake_deg: float = 60.0          # angle between the windshield normal and the optical axis
    radius_mm: float = 4000.0       # radius of curvature of the inner surface (inf = flat)
    distance_mm: float = 60.0       # pupil -> inner surface along the axis
    wedge_mrad: float = 0.0         # rotation of the outer surfaces (HUD wedge)

    def total_thickness(self) -> float:
        return float(sum(t for _, t in self.layers))


def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _refract(d, n, n1, n2):
    """Refract unit directions d at unit normals n (n facing against d) from index n1 to n2."""
    eta = (n1 / n2)[..., None] if np.ndim(n1) else n1 / n2
    cos_i = -np.sum(d * n, axis=-1, keepdims=True)
    sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
    ok = sin2_t[..., 0] < 1.0
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    t = eta * d + (eta * cos_i - cos_t) * n
    return _normalize(t), ok


class _Surface:
    """Sphere (center, radius) or plane (point, normal) with an orientation used to pick the
    intersection on the camera side."""

    def __init__(self, point, normal, radius):
        self.point = np.asarray(point, float)
        self.normal = _normalize(np.asarray(normal, float))  # at `point`, pointing towards the scene
        self.radius = radius
        if np.isfinite(radius):
            # Centre on the camera side for positive radius (windshield convex towards the scene).
            self.center = self.point - self.normal * radius

    def intersect(self, o, d):
        if not np.isfinite(self.radius):
            denom = d @ self.normal
            t = ((self.point - o) @ self.normal) / denom
            return t, np.broadcast_to(self.normal, o.shape)
        oc = o - self.center
        b = np.sum(oc * d, axis=-1)
        c = np.sum(oc * oc, axis=-1) - self.radius ** 2
        disc = b * b - c
        sq = np.sqrt(np.clip(disc, 0.0, None))
        # Rays come from the scene (outside, far side) or travel inside the shell towards the
        # camera; the relevant root is the one nearest to the surface point along the ray.
        t1, t2 = -b - sq, -b + sq
        p1, p2 = o + t1[..., None] * d, o + t2[..., None] * d
        use1 = np.linalg.norm(p1 - self.point, axis=-1) < np.linalg.norm(p2 - self.point, axis=-1)
        t = np.where(use1, t1, t2)
        p = o + t[..., None] * d
        n = _normalize(p - self.center)
        return t, n


def _surfaces(ws: Windshield):
    """Surfaces from the scene side (outer) to the camera side (inner), with media between them."""
    rake = np.radians(ws.rake_deg)
    # Normal at the axis crossing: axis +z tilted about +x so that the top leans towards the camera.
    n0 = np.array([0.0, np.sin(rake), np.cos(rake)])
    p_inner = np.array([0.0, 0.0, ws.distance_mm])
    surfs = []
    offset = 0.0
    boundaries = [0.0]
    for _, t in ws.layers:
        offset += t
        boundaries.append(offset)
    wedge = ws.wedge_mrad * 1e-3
    for k, off in enumerate(boundaries):
        # Distance along the normal from the inner surface; the wedge tilts outer surfaces progressively.
        frac = off / boundaries[-1] if boundaries[-1] > 0 else 0.0
        a = wedge * frac
        c, s = np.cos(a), np.sin(a)
        nk = np.array([0.0, n0[1] * c + n0[2] * s, -n0[1] * s + n0[2] * c])  # rotate about +x
        pk = p_inner + n0 * off
        r = ws.radius_mm + off if np.isfinite(ws.radius_mm) else np.inf
        surfs.append(_Surface(pk, nk, r))
    surfs = surfs[::-1]  # outer first
    media = ["air"] + [g for g, _ in ws.layers[::-1]] + ["air"]
    return surfs, media


def trace_plane_wave(ws: Windshield, scene_dir, wavelength_nm: float, pupil_diameter_mm: float, n_rays: int = 64):
    """Traces a plane wave arriving from `scene_dir` (unit, camera frame) to the pupil plane.

    Returns (x, y, opl, final_dir) for rays that land inside the pupil disk (+ margin).
    """
    s = _normalize(np.asarray(scene_dir, float))
    d0 = -s
    surfs, media = _surfaces(ws)
    idx = [float(glass.ior(m, wavelength_nm)) for m in media]
    # Start plane far on the scene side, perpendicular to d0, centred on the line hitting the pupil.
    e1 = _normalize(np.cross(d0, [0.0, 1.0, 0.0]) if abs(d0[1]) < 0.9 else np.cross(d0, [1.0, 0.0, 0.0]))
    e2 = np.cross(d0, e1)
    far = 10.0 * (ws.distance_mm + ws.total_thickness() + 100.0)
    R = 0.5 * pupil_diameter_mm

    def run(center, half):
        u = np.linspace(-half, half, n_rays)
        a, b = np.meshgrid(u, u)
        o = center + a[..., None] * e1 + b[..., None] * e2
        d = np.broadcast_to(d0, o.shape).copy()
        opl = np.zeros(o.shape[:2])
        alive = np.ones(o.shape[:2], bool)
        for k, srf in enumerate(surfs):
            t, n = srf.intersect(o, d)
            alive &= t > 0
            o = o + t[..., None] * d
            opl += idx[k] * t
            nn = np.where((np.sum(n * d, axis=-1) > 0)[..., None], -n, n)
            d, ok = _refract(d, nn, idx[k], idx[k + 1])
            alive &= ok
        t = -o[..., 2] / d[..., 2]  # pupil plane z = 0
        o = o + t[..., None] * d
        opl += idx[-1] * t
        return o, opl, d, alive

    # Pass 1: chief ray landing point for the centred bundle -> recentre so the bundle covers the pupil.
    c0 = -d0 * far
    o, _, _, alive = run(c0, R * 1.2)
    mid = n_rays // 2
    shift = o[mid, mid] if alive[mid, mid] else np.zeros(3)
    c1 = c0 - (e1 * (shift @ e1) + e2 * (shift @ e2))
    o, opl, d, alive = run(c1, R * 1.3)
    inside = alive & (o[..., 0] ** 2 + o[..., 1] ** 2 <= (1.05 * R) ** 2)
    return o[..., 0][inside], o[..., 1][inside], opl[inside], d[inside]


def pupil_wavefront(ws: Windshield, scene_dir, wavelength_nm, pupil_diameter_mm, n_pupil=64, n_rays=96):
    """Pupil-plane optical path (mm) on an n_pupil x n_pupil grid (row 0 = +y), NaN outside the pupil."""
    x, y, opl, _ = trace_plane_wave(ws, scene_dir, wavelength_nm, pupil_diameter_mm, n_rays)
    gx, gy, mask = pupil_grid(n_pupil)
    R = 0.5 * pupil_diameter_mm
    w = griddata((x, y), opl, (gx * R, gy * R), method="cubic")
    return np.where(mask, w, np.nan), mask


def windshield_package(ws: Windshield, focal_length_mm: float, f_number: float, wavelengths_nm: Sequence[float],
                       fields_deg: Sequence[Tuple[float, float]], psf_pixel_um: float = 0.5, psf_size: int = 64,
                       reference_nm: float = 550.0, refocus: bool = True, n_pupil: int = 64,
                       pad: int = 4) -> OpticsPackage:
    """PSFs of an ideal lens (f, F#) behind the windshield, per wavelength and field.

    Removed terms: piston and tilt at `reference_nm` (per field; the renderer already refracts rays
    through the windshield geometry) -- chromatic tilt differences are kept. With `refocus`, the
    on-axis defocus at `reference_nm` is removed from every field/wavelength (camera focused
    through the windshield, as in vehicle calibration).
    """
    D = focal_length_mm / f_number
    gx, gy, mask = pupil_grid(n_pupil)
    fields = np.asarray(fields_deg, float)
    wls = np.asarray(wavelengths_nm, float)

    def scene_dir(fx, fy):
        v = np.array([np.tan(np.radians(fx)), np.tan(np.radians(fy)), 1.0])
        return v / np.linalg.norm(v)

    defocus = 0.0
    if refocus:
        w0, _ = pupil_wavefront(ws, scene_dir(0.0, 0.0), reference_nm, D, n_pupil)
        _, c = fit_remove(np.nan_to_num(w0), gx, gy, mask, terms=("piston", "tilt", "defocus"))
        defocus = c[-1]
    psf = np.zeros((len(wls), len(fields), psf_size, psf_size))
    for fi, (fx, fy) in enumerate(fields):
        s = scene_dir(fx, fy)
        wref, _ = pupil_wavefront(ws, s, reference_nm, D, n_pupil)
        _, cref = fit_remove(np.nan_to_num(wref), gx, gy, mask, terms=("piston", "tilt"))
        ref_plane = cref[0] + cref[1] * gx + cref[2] * gy
        for li, wl in enumerate(wls):
            w, _ = pupil_wavefront(ws, s, wl, D, n_pupil)
            w = np.nan_to_num(w) - ref_plane - defocus * (2 * (gx * gx + gy * gy) - 1)
            w, _ = fit_remove(w, gx, gy, mask, terms=("piston",))
            waves = w / (wl * 1e-6)
            p, px = psf_from_wavefront(waves, mask, wl, D, focal_length_mm, pad)
            psf[li, fi] = resample_psf(p, px, psf_pixel_um, psf_size)
    meta = {"source": "windshield", "windshield": ws.__dict__, "reference_nm": reference_nm, "refocus": refocus}
    return OpticsPackage(wls, fields, psf, psf_pixel_um, focal_length_mm, f_number, metadata=meta)
