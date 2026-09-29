"""Combining a lens package (e.g. vendor data) with a windshield package.

The windshield package holds PSFs of an *ideal* lens behind the windshield, so they include the
diffraction of that ideal lens. Convolving them with a real lens PSF would count diffraction twice.
We therefore convolve the lens PSF with the windshield's *aberration kernel*: the windshield PSF
deconvolved (Wiener, in the frequency domain) by the ideal-lens PSF of the same aperture. When the
windshield is aberration-free this kernel is a delta and the lens PSF is returned unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

from .package import OpticsPackage
from .pupil import ideal_psf


def aberration_kernel(ws_psf: np.ndarray, ideal: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    Fw = np.fft.fft2(np.fft.ifftshift(ws_psf))
    Fi = np.fft.fft2(np.fft.ifftshift(ideal))
    # Only frequencies passed by the ideal lens carry information; elsewhere use the ideal OTF ratio 1.
    H = np.where(np.abs(Fi) > eps, Fw * np.conj(Fi) / (np.abs(Fi) ** 2 + eps ** 2), 1.0)
    k = np.real(np.fft.fftshift(np.fft.ifft2(H)))
    k = np.clip(k, 0.0, None)
    return k / k.sum()


def combine_packages(lens: OpticsPackage, ws: OpticsPackage) -> OpticsPackage:
    if not np.isclose(lens.psf_pixel_um, ws.psf_pixel_um):
        raise ValueError("resample packages to a common psf_pixel_um first")
    K = lens.psf.shape[-1]
    out = np.zeros_like(lens.psf)
    for li, wl in enumerate(lens.wavelengths_nm):
        ideal = ideal_psf(wl, ws.f_number, ws.focal_length_mm, ws.psf_pixel_um, ws.psf.shape[-1])
        for fi, (fx, fy) in enumerate(lens.fields_deg):
            j = int(np.argmin(np.hypot(ws.fields_deg[:, 0] - fx, ws.fields_deg[:, 1] - fy)))
            kern = aberration_kernel(ws.psf_at(wl, j), ideal)
            c = fftconvolve(lens.psf[li, fi], kern, mode="same")
            c = np.clip(c[:K, :K], 0.0, None)
            out[li, fi] = c / c.sum()
    meta = {"source": "combined", "lens": lens.metadata, "windshield": ws.metadata}
    return OpticsPackage(lens.wavelengths_nm, lens.fields_deg, out, lens.psf_pixel_um, lens.focal_length_mm,
                         lens.f_number, lens.distortion_field_deg, lens.distortion_height_mm, lens.ri_field_deg,
                         lens.ri, lens.transmission_nm, lens.transmission, meta)
