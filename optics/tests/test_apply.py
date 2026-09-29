import numpy as np

from spectral_optics.apply import apply_optics
from spectral_optics.package import OpticsPackage
from spectral_optics.pupil import ideal_psf


def test_single_psf_blurs_a_point_and_conserves_energy():
    H, W, B = 64, 64, 3
    img = np.zeros((H, W, B))
    img[32, 32, :] = 1.0
    wl = np.array([450.0, 550.0, 650.0])
    f, N = 6.0, 8.0
    psf = np.stack([ideal_psf(w, N, f, 0.5, 64) for w in wl])[:, None]
    pkg = OpticsPackage(wl, [(0.0, 0.0)], psf, 0.5, f, N)
    tan_hx = 32 * 3e-3 / f  # 3 um pixels
    out = apply_optics(img, wl, tan_hx, tan_hx, pkg)
    assert np.allclose(out.sum(axis=(0, 1)), 1.0, atol=1e-4)
    # Blue PSF is narrower than red.
    assert out[32, 32, 0] > out[32, 32, 2]


def test_distortion_moves_content_inward_for_barrel():
    H = W = 101
    img = np.zeros((H, W, 1))
    img[50, 90, 0] = 1.0  # point at the right
    f = 6.0
    pitch = 3e-3
    tan_h = (W / 2) * pitch / f
    th_max = np.degrees(np.arctan(tan_h * 1.5))
    th = np.linspace(0, th_max, 50)
    h = f * np.tan(np.radians(th)) * (1 - 0.1 * (th / th_max) ** 2)  # barrel
    psf = np.zeros((1, 1, 5, 5))
    psf[0, 0, 2, 2] = 1
    pkg = OpticsPackage([550.0], [(0, 0)], psf, 0.1, f, 2.0, distortion_field_deg=th, distortion_height_mm=h)
    out = apply_optics(img, np.array([550.0]), tan_h, tan_h, pkg)
    col = np.argmax(out[50, :, 0])
    assert 50 < col < 90


def test_irradiance_camera_equation_on_axis():
    img = np.ones((9, 9, 1))
    psf = np.zeros((1, 1, 3, 3))
    psf[0, 0, 1, 1] = 1
    pkg = OpticsPackage([550.0], [(0, 0)], psf, 1.0, 10.0, 2.0)
    out = apply_optics(img, np.array([550.0]), 0.01, 0.01, pkg, irradiance=True)
    assert np.isclose(out[4, 4, 0], np.pi / (4 * 2.0 ** 2), rtol=1e-4)
