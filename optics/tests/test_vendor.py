import json

import numpy as np

from spectral_optics.package import OpticsPackage
from spectral_optics.vendor import (build_package_from_manifest, distortion_table, expand_rotational, load_psf_text,
                                    mtf_to_psf)


def test_gaussian_mtf_gives_gaussian_psf():
    sigma_um = 3.0
    f = np.linspace(0, 400, 401)  # cycles/mm
    mtf = np.exp(-2 * (np.pi * sigma_um * 1e-3 * f) ** 2)
    psf = mtf_to_psf(f, mtf, mtf, psf_pixel_um=0.5, size=128)
    n = psf.shape[0]
    y, x = np.mgrid[0:n, 0:n] - n // 2
    var = ((x * x) * psf).sum() * 0.25
    assert abs(np.sqrt(var) - sigma_um) < 0.05 * sigma_um


def test_zemax_like_psf_text(tmp_path):
    grid = np.zeros((9, 9))
    grid[4, 4] = 1.0
    grid[3, 4] = 0.5
    lines = ["Listing of Huygens PSF Data", "", "File : lens.ZBB", "0.5500 µm at 10.00 (deg).",
             "Data spacing is 0.250 µm.", "Center point is: row 5, column 5", ""]
    lines += ["\t".join(f"{v:.6E}" for v in row) for row in grid]
    p = tmp_path / "psf.txt"
    p.write_text("\n".join(lines), encoding="utf-16")
    psf, px, meta = load_psf_text(str(p))
    assert psf.shape == (9, 9)
    assert px == 0.25
    assert abs(meta["wavelength_nm"] - 550.0) < 1e-6


def test_distortion_percent_to_height():
    th, h = distortion_table([0, 10, 20], [0.0, -1.0, -4.0], 6.0, "percent")
    assert np.isclose(h[1], 6.0 * np.tan(np.radians(10)) * 0.99)


def test_rotational_expansion_orients_psfs():
    K = 21
    psf = np.zeros((1, 2, K, K))
    psf[0, 0, 10, 10] = 1
    psf[0, 1, 10, 10] = 1
    psf[0, 1, 5, 10] = 1  # blob above centre for the +y field
    pkg = OpticsPackage([550.0], [(0.0, 0.0), (0.0, 20.0)], psf, 1.0, 6.0, 2.0)
    g = expand_rotational(pkg, 20.0, n=3)
    fields = [tuple(f) for f in g.fields_deg]
    right = fields.index((20.0, 0.0))
    p = g.psf[0, right]
    y, x = np.unravel_index(np.argsort(p.ravel())[-2:], p.shape)
    # Secondary blob must now be to the right of centre (x > 10, y ~ 10).
    assert x.max() > 13 and abs(y[np.argmax(x)] - 10) <= 1


def test_manifest_build(tmp_path):
    f = np.linspace(0, 300, 61)
    for fld in (0, 15):
        m = np.clip(1 - f / (300 - fld * 5), 0, 1)
        np.savetxt(tmp_path / f"mtf_{fld}.csv", np.stack([f, m, m * 0.9], axis=1), delimiter=",",
                   header="freq,tan,sag")
    np.savetxt(tmp_path / "dist.txt", [[0, 0.0], [15, -2.0], [30, -8.0]])
    np.savetxt(tmp_path / "ri.txt", [[0, 1.0], [30, 0.6]])
    man = {"focal_length_mm": 6.0, "f_number": 1.8, "psf_pixel_um": 0.5, "psf_size": 48,
           "mtf": [{"wavelength_nm": 550, "field_deg": [0, 0], "file": "mtf_0.csv"},
                   {"wavelength_nm": 550, "field_deg": [0, 15], "file": "mtf_15.csv"}],
           "distortion": {"file": "dist.txt", "kind": "percent"},
           "relative_illumination": {"file": "ri.txt"},
           "rotational_symmetry": True, "max_field_deg": 30, "field_grid": 3}
    (tmp_path / "m.json").write_text(json.dumps(man))
    pkg = build_package_from_manifest(str(tmp_path / "m.json"))
    assert pkg.psf.shape == (1, 9, 48, 48)
    assert np.isclose(pkg.relative_illumination(15.0), 0.8)
    pkg.save(str(tmp_path / "p.npz"))
    back = OpticsPackage.load(str(tmp_path / "p.npz"))
    assert np.allclose(back.psf, pkg.psf, atol=1e-6)
    assert back.metadata["source"] == "vendor"
