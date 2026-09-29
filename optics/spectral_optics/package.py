"""Optics package: the data exchanged between optics sources and the image applier."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class OpticsPackage:
    """Per-wavelength, per-field PSFs plus distortion / relative illumination / transmission.

    Attributes:
        wavelengths_nm: (L,) PSF wavelengths.
        fields_deg: (F, 2) field angles (x, y) in degrees, camera frame (+x right, +y up).
        psf: (L, F, K, K) PSFs on the sensor plane, each normalised to sum 1. Row 0 is the top
            of the image (+y field direction), column 0 the left.
        psf_pixel_um: PSF sample spacing on the sensor in micrometres.
        focal_length_mm, f_number: first-order lens data.
        distortion_field_deg / distortion_height_mm: real image height vs field angle (optional).
            If absent the lens is distortion-free (image height f*tan(theta)).
        ri_field_deg / ri: relative illumination vs field angle (optional, 1 on axis).
        transmission_nm / transmission: lens spectral transmittance (optional).
    """

    wavelengths_nm: np.ndarray
    fields_deg: np.ndarray
    psf: np.ndarray
    psf_pixel_um: float
    focal_length_mm: float
    f_number: float
    distortion_field_deg: Optional[np.ndarray] = None
    distortion_height_mm: Optional[np.ndarray] = None
    ri_field_deg: Optional[np.ndarray] = None
    ri: Optional[np.ndarray] = None
    transmission_nm: Optional[np.ndarray] = None
    transmission: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.wavelengths_nm = np.asarray(self.wavelengths_nm, dtype=np.float64)
        self.fields_deg = np.atleast_2d(np.asarray(self.fields_deg, dtype=np.float64))
        self.psf = np.asarray(self.psf, dtype=np.float64)
        L, F = len(self.wavelengths_nm), len(self.fields_deg)
        if self.psf.ndim != 4 or self.psf.shape[:2] != (L, F) or self.psf.shape[2] != self.psf.shape[3]:
            raise ValueError(f"psf must have shape (L={L}, F={F}, K, K), got {self.psf.shape}")
        if np.any(np.diff(self.wavelengths_nm) <= 0):
            raise ValueError("wavelengths must be strictly ascending")
        s = self.psf.sum(axis=(2, 3), keepdims=True)
        if np.any(s <= 0):
            raise ValueError("every PSF must have positive energy")
        self.psf = self.psf / s

    # ------------------------------------------------------------------ geometry
    def image_height_mm(self, theta_deg):
        """Real image height for field angle(s) theta (deg)."""
        theta = np.asarray(theta_deg, dtype=np.float64)
        if self.distortion_field_deg is None:
            return self.focal_length_mm * np.tan(np.radians(theta))
        return np.interp(theta, self.distortion_field_deg, self.distortion_height_mm)

    def field_from_height_mm(self, h_mm):
        """Inverse of image_height_mm (monotonic tables assumed)."""
        h = np.asarray(h_mm, dtype=np.float64)
        if self.distortion_field_deg is None:
            return np.degrees(np.arctan(h / self.focal_length_mm))
        return np.interp(h, self.distortion_height_mm, self.distortion_field_deg)

    def relative_illumination(self, theta_deg):
        if self.ri is None:
            return np.ones_like(np.asarray(theta_deg, dtype=np.float64))
        return np.interp(np.asarray(theta_deg, dtype=np.float64), self.ri_field_deg, self.ri)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str) -> None:
        arrays = dict(wavelengths_nm=self.wavelengths_nm, fields_deg=self.fields_deg,
                      psf=self.psf.astype(np.float32), psf_pixel_um=np.float64(self.psf_pixel_um),
                      focal_length_mm=np.float64(self.focal_length_mm), f_number=np.float64(self.f_number),
                      metadata=np.bytes_(json.dumps(self.metadata)))
        for name in ("distortion_field_deg", "distortion_height_mm", "ri_field_deg", "ri",
                     "transmission_nm", "transmission"):
            v = getattr(self, name)
            if v is not None:
                arrays[name] = np.asarray(v, dtype=np.float64)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "OpticsPackage":
        with np.load(path) as z:
            kw = {k: z[k] for k in z.files if k != "metadata"}
            meta = json.loads(bytes(z["metadata"]).decode()) if "metadata" in z.files else {}
        for k in ("psf_pixel_um", "focal_length_mm", "f_number"):
            kw[k] = float(kw[k])
        return cls(metadata=meta, **kw)

    # ------------------------------------------------------------------ PSF lookup
    def psf_at(self, wavelength_nm: float, field_index: int) -> np.ndarray:
        """PSF for one field, linearly interpolated in wavelength (clamped at the ends)."""
        wl = self.wavelengths_nm
        if wavelength_nm <= wl[0]:
            return self.psf[0, field_index]
        if wavelength_nm >= wl[-1]:
            return self.psf[-1, field_index]
        i = int(np.searchsorted(wl, wavelength_nm)) - 1
        t = (wavelength_nm - wl[i]) / (wl[i + 1] - wl[i])
        p = (1 - t) * self.psf[i, field_index] + t * self.psf[i + 1, field_index]
        return p / p.sum()
