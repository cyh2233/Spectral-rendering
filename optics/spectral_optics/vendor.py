"""Vendor optical performance data -> OpticsPackage.

For a black-box lens (e.g. an encrypted Zemax .ZBB file, not traceable without OpticStudio) the
lens vendor can export performance data from OpticStudio. Supported inputs:

  * PSF grids: Zemax text exports ("Huygens PSF"/"FFT PSF" listings), .npy or .csv matrices
  * MTF curves (tangential/sagittal vs spatial frequency) -> PSF via an elliptical OTF model
    (zero phase: symmetric PSFs; coma-like asymmetry is not recoverable from MTF alone)
  * distortion vs field (percent F-tan(theta) distortion, or real image height)
  * relative illumination vs field
  * spectral transmittance

A JSON manifest ties the files together; see `build_package_from_manifest` and docs/optics_stage.md.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .package import OpticsPackage
from .pupil import resample_psf

_NUM = re.compile(r"^[\s,;]*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?([\s,;]+[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?)*[\s,;]*$")


def _read_text(path: str) -> List[str]:
    for enc in ("utf-8", "utf-16", "latin-1"):  # Zemax text exports are often UTF-16
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            if "\x00" not in text:
                return text.splitlines()
        except UnicodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def _floats(line: str) -> List[float]:
    return [float(t) for t in re.split(r"[\s,;]+", line.strip()) if t]


def load_table(path: str) -> np.ndarray:
    """Numeric rows of a text/CSV file (header and comment lines skipped); rows must agree in length."""
    rows = [_floats(l) for l in _read_text(path) if _NUM.match(l)]
    if not rows:
        raise ValueError(f"no numeric data in {path}")
    n = max(len(r) for r in rows)
    rows = [r for r in rows if len(r) == n]
    return np.array(rows, dtype=np.float64)


def load_psf_text(path: str) -> Tuple[np.ndarray, Optional[float], Dict[str, float]]:
    """Parses a PSF grid export. Returns (psf, pixel_um or None, metadata).

    Recognised metadata lines (case-insensitive): "... spacing is 0.250 um/µm/mm",
    "wavelength ... 0.5500 um/nm", "field ... 10.00 (deg)".
    """
    lines = _read_text(path)
    meta: Dict[str, float] = {}
    pixel_um = None
    for l in lines:
        low = l.lower()
        m = re.search(r"spacing\s+is\s+([-+\d.eE]+)\s*(µm|um|mm|microns?)", low)
        if m:
            v = float(m.group(1))
            pixel_um = v * 1000.0 if m.group(2) == "mm" else v
        m = re.search(r"wavelength[^\d-]*([-+\d.eE]+)\s*(µm|um|nm|microns?)", low)
        if m and "wavelength_nm" not in meta:
            v = float(m.group(1))
            meta["wavelength_nm"] = v if m.group(2) == "nm" else v * 1000.0
        m = re.search(r"([-+\d.eE]+)\s*(µm|um|nm)\s+at\s+([-+\d.eE]+)\s*(?:\((deg|mm)\))?", low)
        if m:  # Zemax listing header: "0.5500 µm at 10.00 (deg)."
            v = float(m.group(1))
            meta.setdefault("wavelength_nm", v if m.group(2) == "nm" else v * 1000.0)
            meta.setdefault("field", float(m.group(3)))
        m = re.search(r"field[^\d-]*([-+\d.eE]+)", low)
        if m and "field" not in meta:
            meta["field"] = float(m.group(1))
    rows = [_floats(l) for l in lines if _NUM.match(l)]
    # The PSF grid is the largest block of rows sharing the same (largest) length.
    n = max(len(r) for r in rows)
    grid = np.array([r for r in rows if len(r) == n], dtype=np.float64)
    if grid.shape[0] != grid.shape[1]:
        raise ValueError(f"PSF grid in {path} is not square: {grid.shape}")
    return grid, pixel_um, meta


def load_psf(path: str, fmt: str = "auto"):
    ext = os.path.splitext(path)[1].lower()
    if fmt == "npy" or (fmt == "auto" and ext == ".npy"):
        return np.load(path).astype(np.float64), None, {}
    if fmt == "csv" or (fmt == "auto" and ext == ".csv"):
        return load_table(path), None, {}
    return load_psf_text(path)


def mtf_to_psf(freq_cyc_mm: Sequence[float], mtf_tangential: Sequence[float], mtf_sagittal: Sequence[float],
               psf_pixel_um: float, size: int) -> np.ndarray:
    """PSF (field along +y) from tangential/sagittal MTF curves via a zero-phase elliptical OTF:
    OTF(nu, phi) = T(nu) cos^2(phi) + S(nu) sin^2(phi), phi measured from the tangential (+y) axis."""
    f = np.asarray(freq_cyc_mm, float)
    T = np.asarray(mtf_tangential, float)
    S = np.asarray(mtf_sagittal, float)
    N = size
    fx = np.fft.fftfreq(N, d=psf_pixel_um * 1e-3)  # cycles per mm
    FX, FY = np.meshgrid(fx, fx)
    nu = np.hypot(FX, FY)
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(nu > 0, (FY / nu) ** 2, 1.0)
    t = np.interp(nu, f, T, right=0.0)
    s = np.interp(nu, f, S, right=0.0)
    otf = t * c2 + s * (1.0 - c2)
    psf = np.real(np.fft.fftshift(np.fft.ifft2(otf)))
    psf = np.clip(psf, 0.0, None)
    return psf / psf.sum()


def rotate_psf(psf: np.ndarray, phi_deg: float) -> np.ndarray:
    """Rotates a PSF defined for a field along +y to a field direction phi (deg, from +y towards +x)."""
    if abs(phi_deg) < 1e-9:
        return psf
    out = ndimage.rotate(psf, -phi_deg, reshape=False, order=1, mode="constant", cval=0.0)
    out = np.clip(out, 0.0, None)
    return out / out.sum()


def expand_rotational(pkg: OpticsPackage, max_field_deg: float, n: int = 5) -> OpticsPackage:
    """Turns PSFs sampled along the +y field axis (rotationally symmetric lens) into an n x n
    rectangular field grid spanning +-max_field_deg in x and y."""
    radial = np.abs(pkg.fields_deg[:, 1]) if np.allclose(pkg.fields_deg[:, 0], 0) else np.hypot(*pkg.fields_deg.T)
    order = np.argsort(radial)
    radial = radial[order]
    grid = np.linspace(-max_field_deg, max_field_deg, n)
    fields = [(x, y) for y in grid[::-1] for x in grid]
    L, K = len(pkg.wavelengths_nm), pkg.psf.shape[-1]
    out = np.zeros((L, len(fields), K, K))
    for fi, (fx, fy) in enumerate(fields):
        r = np.hypot(fx, fy)
        phi = np.degrees(np.arctan2(fx, fy)) if r > 0 else 0.0
        j = int(np.clip(np.searchsorted(radial, r) - 1, 0, max(0, len(radial) - 2)))
        if len(radial) == 1:
            w = [(order[0], 1.0)]
        else:
            t = float(np.clip((r - radial[j]) / (radial[j + 1] - radial[j]), 0.0, 1.0))
            w = [(order[j], 1.0 - t), (order[j + 1], t)]
        for li in range(L):
            p = sum(wt * pkg.psf[li, k] for k, wt in w)
            out[li, fi] = rotate_psf(p / p.sum(), phi)
    meta = dict(pkg.metadata, expanded_rotational=True)
    return OpticsPackage(pkg.wavelengths_nm, np.array(fields), out, pkg.psf_pixel_um, pkg.focal_length_mm,
                         pkg.f_number, pkg.distortion_field_deg, pkg.distortion_height_mm, pkg.ri_field_deg,
                         pkg.ri, pkg.transmission_nm, pkg.transmission, meta)


def distortion_table(field_deg, values, focal_length_mm: float, kind: str = "percent"):
    """Returns (field_deg, real_image_height_mm). kind: 'percent' (F-tan(theta) distortion in %),
    or 'height_mm' (real image height)."""
    th = np.asarray(field_deg, float)
    v = np.asarray(values, float)
    if kind == "percent":
        h = focal_length_mm * np.tan(np.radians(th)) * (1.0 + v / 100.0)
    elif kind == "height_mm":
        h = v
    else:
        raise ValueError("kind must be 'percent' or 'height_mm'")
    o = np.argsort(th)
    return th[o], h[o]


def build_package_from_manifest(path: str) -> OpticsPackage:
    """Builds an OpticsPackage from a vendor-data manifest (JSON), paths relative to the manifest."""
    with open(path) as f:
        m = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: p if os.path.isabs(p) else os.path.join(base, p)  # noqa: E731
    f_mm, fno = float(m["focal_length_mm"]), float(m["f_number"])
    px, size = float(m.get("psf_pixel_um", 0.5)), int(m.get("psf_size", 64))
    entries = []  # (wavelength, (fx, fy), psf)
    for e in m.get("psf", []):
        psf, pix, meta = load_psf(rel(e["file"]), e.get("format", "auto"))
        pix = float(e.get("pixel_um", pix or 0.0))
        if pix <= 0:
            raise ValueError(f"{e['file']}: PSF pixel spacing unknown; add 'pixel_um' to the manifest entry")
        wl = float(e.get("wavelength_nm", meta.get("wavelength_nm", 0.0)))
        fld = e.get("field_deg", [0.0, meta.get("field", 0.0)])
        entries.append((wl, tuple(map(float, fld)), resample_psf(psf / psf.sum(), pix, px, size)))
    for e in m.get("mtf", []):
        tab = load_table(rel(e["file"]))
        cols = e.get("columns", {"freq": 0, "tangential": 1, "sagittal": 2})
        psf = mtf_to_psf(tab[:, cols["freq"]], tab[:, cols["tangential"]], tab[:, cols["sagittal"]], px, size)
        entries.append((float(e["wavelength_nm"]), tuple(map(float, e.get("field_deg", [0.0, 0.0]))), psf))
    if not entries:
        raise ValueError("manifest has no 'psf' or 'mtf' entries")
    wls = sorted({e[0] for e in entries})
    flds = sorted({e[1] for e in entries}, key=lambda t: (t[1], t[0]))
    table = {(e[0], e[1]): e[2] for e in entries}
    psf = np.zeros((len(wls), len(flds), size, size))
    for li, wl in enumerate(wls):
        for fi, fl in enumerate(flds):
            if (wl, fl) not in table:
                raise ValueError(f"missing PSF/MTF for wavelength {wl} nm, field {fl}")
            psf[li, fi] = table[(wl, fl)]
    kw = {}
    if "distortion" in m:
        d = m["distortion"]
        tab = load_table(rel(d["file"]))
        c = d.get("columns", {"field_deg": 0, "value": 1})
        kw["distortion_field_deg"], kw["distortion_height_mm"] = distortion_table(
            tab[:, c["field_deg"]], tab[:, c["value"]], f_mm, d.get("kind", "percent"))
    if "relative_illumination" in m:
        r = m["relative_illumination"]
        tab = load_table(rel(r["file"]))
        c = r.get("columns", {"field_deg": 0, "value": 1})
        o = np.argsort(tab[:, c["field_deg"]])
        kw["ri_field_deg"], kw["ri"] = tab[o, c["field_deg"]], tab[o, c["value"]]
    if "transmission" in m:
        tab = load_table(rel(m["transmission"]["file"]))
        kw["transmission_nm"], kw["transmission"] = tab[:, 0], tab[:, 1]
    pkg = OpticsPackage(np.array(wls), np.array(flds), psf, px, f_mm, fno,
                        metadata={"source": "vendor", "manifest": os.path.abspath(path)}, **kw)
    if m.get("rotational_symmetry", False):
        pkg = expand_rotational(pkg, float(m.get("max_field_deg", np.max(np.abs(pkg.fields_deg)))),
                                int(m.get("field_grid", 5)))
    return pkg
