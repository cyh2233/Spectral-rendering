"""Combining a lens package (e.g. vendor data) with a windshield package.

The windshield package holds PSFs of an *ideal* lens behind the windshield, so they include the
diffraction of that ideal lens. Convolving them with a real lens PSF would count diffraction twice.
We therefore convolve the lens PSF with the windshield's *aberration kernel*: the windshield PSF
deconvolved (Wiener, in the frequency domain) by the ideal-lens PSF of the same aperture. When the
windshield is aberration-free this kernel is a delta and the lens PSF is returned unchanged.
"""
from __future__ import annotations

import numpy as np

from .package import OpticsPackage
from .pupil import ideal_psf


def aberration_kernel(ws_psf: np.ndarray, ideal: np.ndarray, rel_threshold: float = 3e-2) -> np.ndarray:
    """OTF ratio OTF_ws / OTF_ideal where the ideal OTF carries information (|OTF| above
    rel_threshold of its DC value); elsewhere 1. Both PSFs are centred at index N // 2."""
    Fw = np.fft.fft2(np.fft.ifftshift(ws_psf))
    Fi = np.fft.fft2(np.fft.ifftshift(ideal))
    ok = np.abs(Fi) > rel_threshold * np.abs(Fi[0, 0])
    H = np.ones_like(Fw)
    H[ok] = Fw[ok] / Fi[ok]
    k = np.real(np.fft.fftshift(np.fft.ifft2(H)))
    k = np.clip(k, 0.0, None)
    return k / k.sum()


def _convolve_centred(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution of two same-size PSFs centred at N // 2 (result centred likewise)."""
    c = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(a)) * np.fft.fft2(np.fft.ifftshift(b)))))
    c = np.clip(c, 0.0, None)
    return c / c.sum()


def combine_packages(lens: OpticsPackage, ws: OpticsPackage) -> OpticsPackage:
    if not np.isclose(lens.psf_pixel_um, ws.psf_pixel_um):
        raise ValueError("resample packages to a common psf_pixel_um first")
    out = np.zeros_like(lens.psf)
    for li, wl in enumerate(lens.wavelengths_nm):
        # Same pupil sampling as the windshield model, so an aberration-free windshield gives a delta.
        ideal = ideal_psf(wl, ws.f_number, ws.focal_length_mm, ws.psf_pixel_um, ws.psf.shape[-1],
                          n_pupil=int(ws.metadata.get("n_pupil", 64)), pad=int(ws.metadata.get("pad", 4)))
        for fi, (fx, fy) in enumerate(lens.fields_deg):
            j = int(np.argmin(np.hypot(ws.fields_deg[:, 0] - fx, ws.fields_deg[:, 1] - fy)))
            kern = aberration_kernel(ws.psf_at(wl, j), ideal)
            if kern.shape != lens.psf.shape[2:]:
                raise ValueError("lens and windshield packages need the same psf_size")
            out[li, fi] = _convolve_centred(lens.psf[li, fi], kern)
    meta = {"source": "combined", "lens": lens.metadata, "windshield": ws.metadata}
    return OpticsPackage(lens.wavelengths_nm, lens.fields_deg, out, lens.psf_pixel_um, lens.focal_length_mm,
                         lens.f_number, lens.distortion_field_deg, lens.distortion_height_mm, lens.ri_field_deg,
                         lens.ri, lens.transmission_nm, lens.transmission, meta)
