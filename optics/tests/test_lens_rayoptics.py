import numpy as np
import pytest

ro = pytest.importorskip("rayoptics")

from spectral_optics.lens_rayoptics import lens_package  # noqa: E402


def singlet():
    import rayoptics.elem.elements  # noqa: F401  (avoids a circular import in rayoptics 0.9.5)
    from rayoptics.optical.opticalmodel import OpticalModel
    from rayoptics.raytr.opticalspec import FieldSpec, PupilSpec, WvlSpec
    opm = OpticalModel()
    sm, osp = opm['seq_model'], opm['optical_spec']
    osp['pupil'] = PupilSpec(osp, key=['object', 'epd'], value=12.5)
    osp['fov'] = FieldSpec(osp, key=['object', 'angle'], value=5., flds=[0., 1.0], is_relative=True)
    osp['wvls'] = WvlSpec([(550., 1.)], ref_wl=0)
    opm.radius_mode = True
    sm.gaps[0].thi = 1e10
    sm.add_surface([51.68, 4.0, 'N-BK7', 'Schott'])
    sm.set_stop()
    sm.add_surface([0., 96.0])
    opm.update_model()
    return opm


def test_coma_orientation_matches_spot_diagram():
    pkg = lens_package(singlet(), [550.0], [0.0, 5.0], psf_pixel_um=1.0, psf_size=96, n_pupil=64)
    p = pkg.psf[0, 1]
    n = p.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    cy = ((n // 2 - y) * p).sum()
    # RayOptics spot centroid for this field is ~ -4.7 um in y (towards the axis).
    assert -6.5 < cy < -3.0
    assert pkg.focal_length_mm == pytest.approx(99.67, rel=1e-3)
    assert pkg.distortion_height_mm[-1] > 0
