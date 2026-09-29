import numpy as np
import pytest

from spectral_optics import glass
from spectral_optics.pupil import ideal_psf
from spectral_optics.windshield import Windshield, trace_plane_wave, windshield_package


def centroid(psf, px):
    n = psf.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    return ((x - n // 2) * psf).sum() * px, ((n // 2 - y) * psf).sum() * px


def test_flat_parallel_plate_is_aberration_free():
    ws = Windshield(layers=[("soda_lime", 5.0)], rake_deg=60.0, radius_mm=np.inf, distance_mm=40.0)
    pkg = windshield_package(ws, 6.0, 2.0, [550.0], [(0.0, 0.0), (10.0, 5.0)], psf_pixel_um=0.25, psf_size=48,
                             refocus=False)
    ideal = ideal_psf(550.0, 2.0, 6.0, 0.25, 48)
    for fi in range(2):
        assert np.abs(pkg.psf[0, fi] - ideal).max() < 0.02 * ideal.max()


def test_plane_wave_through_flat_plate_keeps_direction():
    ws = Windshield(layers=[("soda_lime", 5.0)], rake_deg=50.0, radius_mm=np.inf)
    s = np.array([0.1, 0.05, 1.0])
    s /= np.linalg.norm(s)
    _, _, _, d = trace_plane_wave(ws, s, 600.0, 4.0, 16)
    assert np.allclose(-d.mean(axis=0), s, atol=1e-9)


def test_wedge_produces_lateral_colour_of_the_right_size_and_sign():
    alpha = 20e-3  # rad
    ws = Windshield(layers=[("soda_lime", 5.0)], rake_deg=0.0, radius_mm=np.inf, wedge_mrad=alpha * 1e3)
    f = 50.0
    pkg = windshield_package(ws, f, 8.0, [450.0, 650.0], [(0.0, 0.0)], psf_pixel_um=1.0, psf_size=128,
                             reference_nm=550.0, refocus=False)
    cx_b, cy_b = centroid(pkg.psf[0, 0], 1.0)
    cx_r, cy_r = centroid(pkg.psf[1, 0], 1.0)
    # Thin prism: deviation (n - 1) * alpha -> image separation f * (n_450 - n_650) * alpha.
    dn = glass.ior("soda_lime", 450.0) - glass.ior("soda_lime", 650.0)
    expect = f * dn * alpha * 1e3  # um
    sep = cy_b - cy_r
    assert abs(abs(sep) - expect) < 0.15 * expect
    assert abs(cx_b - cx_r) < 0.1 * expect
    # Sign: blue is deviated further in the same direction as the geometric deviation.
    s = np.array([0.0, 0.0, 1.0])
    _, _, _, d = trace_plane_wave(ws, s, 550.0, 6.0, 8)
    apparent = -d.mean(axis=0)
    assert np.sign(sep) == np.sign(apparent[1] - s[1])


def test_curved_raked_windshield_is_astigmatic_off_focus():
    ws = Windshield(layers=[("soda_lime", 5.0)], rake_deg=65.0, radius_mm=300.0, distance_mm=50.0)
    pkg = windshield_package(ws, 25.0, 2.0, [550.0], [(0.0, 0.0)], psf_pixel_um=0.5, psf_size=96, refocus=False)
    p = pkg.psf[0, 0]
    n = p.shape[0]
    y, x = np.mgrid[0:n, 0:n] - n // 2
    sxx, syy = (x * x * p).sum(), (y * y * p).sum()
    assert max(sxx, syy) / min(sxx, syy) > 1.3  # different tangential / sagittal blur
