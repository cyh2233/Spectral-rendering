"""Lens prescriptions (.zmx / CODE V .seq) -> OpticsPackage via RayOptics (open source).

Only *unencrypted* prescriptions can be traced. Zemax black-box (.ZBB) files require OpticStudio;
for those use vendor performance data (spectral_optics.vendor).

PSFs are computed by FFT of the RayOptics wavefront (OPD in waves, referenced to the chief-ray
image point) on the exit pupil. Orientation: validated against RayOptics spot diagrams (coma
centroid direction), see tests/test_lens_rayoptics.py.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .package import OpticsPackage
from .pupil import psf_from_wavefront, resample_psf


def _ro():
    from rayoptics.optical.opticalmodel import OpticalModel  # noqa: F401  (import check)
    from rayoptics.raytr import analyses, trace
    from rayoptics.raytr.opticalspec import FieldSpec, WvlSpec
    return analyses, trace, FieldSpec, WvlSpec


def load_prescription(path: str):
    """Reads a Zemax .zmx or CODE V .seq file into a RayOptics OpticalModel."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".zmx":
        from rayoptics.zemax import zmxread
        opm, _ = zmxread.read_lens_file(path)
    elif ext == ".seq":
        from rayoptics.codev import cmdproc
        opm, _ = cmdproc.read_lens(path)
    elif ext == ".zbb":
        raise ValueError("Zemax black-box (.ZBB) files are encrypted and need OpticStudio; "
                         "export PSF/MTF/distortion data instead (spectral_optics.vendor)")
    else:
        raise ValueError(f"unsupported prescription format '{ext}' (use .zmx or .seq)")
    opm.update_model()
    return opm


def set_fields(opm, field_angles_deg: Sequence[float]):
    _, _, FieldSpec, _ = _ro()
    osp = opm['optical_spec']
    fmax = max(max(abs(a) for a in field_angles_deg), 1e-6)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], value=fmax,
                           flds=[a / fmax for a in field_angles_deg], is_relative=True)
    opm.update_model()


def lens_package(opm, wavelengths_nm: Sequence[float], field_angles_deg: Sequence[float],
                 psf_pixel_um: float = 0.5, psf_size: int = 64, n_pupil: int = 64, pad: int = 4,
                 distortion_samples: int = 16) -> OpticsPackage:
    """PSFs for fields along +y (rotationally symmetric lens) + distortion table from chief rays.
    Expand to a 2-D field grid with vendor.expand_rotational before applying to images."""
    analyses, trace, _, WvlSpec = _ro()
    wls = [float(w) for w in wavelengths_nm]
    osp = opm['optical_spec']
    osp['wvls'] = WvlSpec([(w, 1.0) for w in wls], ref_wl=len(wls) // 2)
    angles = [float(a) for a in field_angles_deg]
    set_fields(opm, angles)
    fod = opm['analysis_results']['parax_data'].fod
    psf = np.zeros((len(wls), len(angles), psf_size, psf_size))
    for fi in range(len(angles)):
        fld = osp['fov'].fields[fi]
        for li, wl in enumerate(wls):
            rg = analyses.RayGrid(opm, f=fi, wl=wl, num_rays=n_pupil)
            opd = rg.grid[2].T[::-1, :]  # RayGrid is indexed [x, y]; pupil arrays are row 0 = +y
            mask = ~np.isnan(opd)
            xs = rg.grid[0][:, 0]
            span = (xs[-1] - xs[0]) * n_pupil / (n_pupil - 1)  # normalised pupil width covered
            D = fod.exp_radius * span
            p, px = psf_from_wavefront(np.nan_to_num(opd), mask, wl, D, abs(fld.ref_sphere[2]), pad)
            psf[li, fi] = resample_psf(p, px, psf_pixel_um, psf_size)
    # Distortion from real chief rays at the central wavelength.
    fmax = max(abs(a) for a in angles) if max(abs(a) for a in angles) > 0 else 1.0
    th = np.linspace(0.0, fmax, distortion_samples)
    set_fields(opm, list(th))
    heights = []
    for fi in range(len(th)):
        cr, err = trace.trace_safe(opm, [0.0, 0.0], osp['fov'].fields[fi], wls[len(wls) // 2], None, None)
        heights.append(abs(cr[0][-1][0][1]) if err is None else np.nan)
    heights = np.array(heights)
    ok = ~np.isnan(heights)
    set_fields(opm, angles)
    fields = np.array([[0.0, a] for a in angles])
    return OpticsPackage(np.array(wls), fields, psf, psf_pixel_um, float(fod.efl), float(fod.efl / (2 * fod.enp_radius)),
                         distortion_field_deg=th[ok], distortion_height_mm=heights[ok],
                         metadata={"source": "rayoptics"})
