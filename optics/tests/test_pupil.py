import numpy as np

from spectral_optics.pupil import airy_first_zero_um, ideal_psf, psf_from_wavefront, pupil_grid, resample_psf


def radial_profile(psf, px):
    n = psf.shape[0]
    c = n // 2
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - c, y - c) * px
    bins = np.arange(0, r.max(), px)
    idx = np.digitize(r.ravel(), bins)
    prof = np.bincount(idx, psf.ravel()) / np.maximum(np.bincount(idx), 1)
    return bins, prof[1:len(bins) + 1]


def test_airy_first_zero():
    wl, N, f = 550.0, 4.0, 10.0
    psf = ideal_psf(wl, N, f, target_pixel_um=0.1, size=201, n_pupil=128, pad=8)
    r, prof = radial_profile(psf, 0.1)
    z = airy_first_zero_um(wl, N)
    near = (r > 0.7 * z) & (r < 1.3 * z)
    r_min = r[near][np.argmin(prof[near])]
    assert abs(r_min - z) < 0.12 * z
    assert prof[0] > 50 * prof[near].min()


def test_tilt_moves_psf_up_for_light_from_above():
    # Plane wave from apparent direction +y (s_y = a): W = -(s . p) / lambda  -> image above centre.
    n, wl, f, D = 64, 500.0, 20.0, 5.0
    x, y, mask = pupil_grid(n)
    a = 2e-4  # rad
    W = -(a * y * D / 2) / (wl * 1e-6)
    psf, px = psf_from_wavefront(W, mask, wl, D, f, pad=8)
    N = psf.shape[0]
    yy, xx = np.mgrid[0:N, 0:N]
    cy = ((N // 2 - yy) * psf).sum() * px
    cx = ((xx - N // 2) * psf).sum() * px
    assert abs(cy - a * f * 1e3) < 0.05 * a * f * 1e3
    assert abs(cx) < 0.05


def test_resample_preserves_energy_and_centre():
    psf = ideal_psf(600.0, 2.0, 5.0, 0.25, 65)
    r = resample_psf(psf, 0.25, 1.0, 17)
    assert np.isclose(r.sum(), 1.0)
    assert np.unravel_index(np.argmax(r), r.shape) == (8, 8)
