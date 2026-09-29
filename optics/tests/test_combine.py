import numpy as np

from spectral_optics.combine import combine_packages
from spectral_optics.package import OpticsPackage
from spectral_optics.pupil import ideal_psf
from spectral_optics.windshield import Windshield, windshield_package


def lens_pkg(f=25.0, N=2.0):
    wl = [550.0]
    psf = np.stack([ideal_psf(550.0, N * 1.5, f, 0.5, 64)])[:, None]  # a softer "vendor" lens
    return OpticsPackage(wl, [(0.0, 0.0)], psf, 0.5, f, N)


def second_moment(p):
    n = p.shape[0]
    y, x = np.mgrid[0:n, 0:n] - n // 2
    return (x * x * p).sum() + (y * y * p).sum()


def test_flat_windshield_leaves_lens_psf_unchanged():
    lens = lens_pkg()
    ws = windshield_package(Windshield(layers=[("soda_lime", 5.0)], radius_mm=np.inf), 25.0, 2.0, [550.0],
                            [(0.0, 0.0)], psf_pixel_um=0.5, psf_size=64, refocus=False)
    c = combine_packages(lens, ws)
    assert np.abs(c.psf[0, 0] - lens.psf[0, 0]).max() < 0.05 * lens.psf.max()


def test_aberrated_windshield_broadens_lens_psf():
    lens = lens_pkg()
    ws = windshield_package(Windshield(layers=[("soda_lime", 5.0)], rake_deg=65.0, radius_mm=300.0), 25.0, 2.0,
                            [550.0], [(0.0, 0.0)], psf_pixel_um=0.5, psf_size=64, refocus=False)
    c = combine_packages(lens, ws)
    assert second_moment(c.psf[0, 0]) > 1.2 * second_moment(lens.psf[0, 0])
