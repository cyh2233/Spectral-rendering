"""Optics stage for the spectral renderer.

The renderer produces a pinhole spectral radiance cube. This package turns it into what a real
camera behind a windshield sees, using an *optics package*: per-wavelength, per-field PSFs,
lens distortion, relative illumination and spectral transmission.

Sources of optics packages:
  * vendor data for a black-box lens (PSF / MTF / distortion / RI exports) -- spectral_optics.vendor
  * lens prescriptions (.zmx / .seq) traced with RayOptics            -- spectral_optics.lens_rayoptics
  * an ideal (diffraction-limited) lens                                -- spectral_optics.pupil
  * a windshield (tilted, curved, laminated, wedged glass)             -- spectral_optics.windshield

Double-counting rule: the renderer already models the windshield's geometric refraction
(ray displacement / distortion), Fresnel ghosts and spectral transmittance. The windshield model
here therefore contributes PSF *shape* only: piston and the reference-wavelength tilt are removed,
while chromatic tilt differences (lateral colour from a wedge) and astigmatism/defocus remain.
"""
from .package import OpticsPackage  # noqa: F401

__version__ = "0.1.0"
