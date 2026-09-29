"""Fourier optics: PSFs from pupil-plane wavefronts, resampling, ideal lenses."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def pupil_grid(n: int):
    """Normalised pupil coordinates (-1..1) sampled at n x n cell centres and the unit-disk mask."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    x, y = np.meshgrid(c, -c)  # row 0 = +y (top)
    return x, y, (x * x + y * y) <= 1.0


def psf_from_wavefront(opd_waves: np.ndarray, mask: np.ndarray, wavelength_nm: float, pupil_diameter_mm: float,
                       focal_length_mm: float, pad: int = 4):
    """Incoherent PSF |FT(P exp(2 pi i W))|^2 of an n x n pupil sampling.

    Returns (psf, pixel_um): psf normalised to sum 1, row 0 = top of the image.
    Image-plane sampling: dx = lambda f / (N_fft * pupil_sample_spacing).
    """
    n = opd_waves.shape[0]
    N = n * pad
    field = np.zeros((N, N), dtype=np.complex128)
    w = np.where(mask, np.nan_to_num(opd_waves), 0.0)
    field[:n, :n] = mask * np.exp(2j * np.pi * w)
    F = np.fft.fftshift(np.fft.fft2(field))
    psf = np.abs(F) ** 2
    # Convention: `opd_waves` is optical path length (increasing along propagation) on the pupil
    # plane, row 0 = +y. A plane wave arriving from apparent direction s has W = -(s . p) / lambda.
    # The returned PSF is in the upright image convention used by the renderer (+x right, +y up,
    # row 0 = top): light from direction +s_y lands above the centre. With numpy's FFT sign that
    # requires flipping both axes; the roll keeps the zero-frequency sample at index N // 2.
    psf = np.roll(psf[::-1, ::-1], (1, 1), axis=(0, 1))
    dx_pupil_mm = pupil_diameter_mm / n
    pixel_um = wavelength_nm * 1e-6 * focal_length_mm / (N * dx_pupil_mm) * 1e3
    return psf / psf.sum(), pixel_um


def resample_psf(psf: np.ndarray, pixel_um: float, target_pixel_um: float, size: int) -> np.ndarray:
    """Resamples a centred PSF onto a size x size grid of spacing target_pixel_um (energy preserving)."""
    n = psf.shape[0]
    c_src = n // 2
    c_dst = size // 2
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    src_y = (yy - c_dst) * target_pixel_um / pixel_um + c_src
    src_x = (xx - c_dst) * target_pixel_um / pixel_um + c_src
    if target_pixel_um > pixel_um * 1.5:
        # Down-sampling: pre-filter to avoid aliasing (box of the target pixel footprint).
        k = target_pixel_um / pixel_um
        psf = ndimage.uniform_filter(psf, size=max(1, int(round(k))), mode="constant")
    out = ndimage.map_coordinates(psf, [src_y, src_x], order=1, mode="constant", cval=0.0)
    out = np.clip(out, 0.0, None)
    s = out.sum()
    if s <= 0:
        raise ValueError("PSF resampling lost all energy (PSF larger than the target grid?)")
    return out / s


def ideal_psf(wavelength_nm: float, f_number: float, focal_length_mm: float, target_pixel_um: float, size: int,
              n_pupil: int = 128, pad: int = 4, opd_waves: np.ndarray | None = None) -> np.ndarray:
    """Diffraction-limited (optionally aberrated) PSF of a lens with a circular pupil."""
    x, y, mask = pupil_grid(n_pupil)
    w = np.zeros_like(x) if opd_waves is None else opd_waves
    psf, px = psf_from_wavefront(w, mask, wavelength_nm, focal_length_mm / f_number, focal_length_mm, pad)
    return resample_psf(psf, px, target_pixel_um, size)


def airy_first_zero_um(wavelength_nm: float, f_number: float) -> float:
    return 1.2197 * wavelength_nm * 1e-3 * f_number


def fit_remove(opd: np.ndarray, x: np.ndarray, y: np.ndarray, mask: np.ndarray, terms=("piston", "tilt")):
    """Least-squares removal of piston / tilt / defocus over the pupil mask. Returns (residual, coeffs)."""
    cols = []
    if "piston" in terms:
        cols.append(np.ones_like(x))
    if "tilt" in terms:
        cols += [x, y]
    if "defocus" in terms:
        cols.append(2 * (x * x + y * y) - 1)
    A = np.stack([c[mask] for c in cols], axis=1)
    coef, *_ = np.linalg.lstsq(A, opd[mask], rcond=None)
    fit = sum(c * k for c, k in zip(cols, coef))
    return np.where(mask, opd - fit, 0.0), coef
