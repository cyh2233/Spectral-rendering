"""Renderer (C++) -> NPZ -> numpy -> optics stage. Skipped if the renderer is not built."""
import json
import os
import subprocess

import numpy as np
import pytest

from spectral_optics.apply import apply_to_npz
from spectral_optics.package import OpticsPackage
from spectral_optics.pupil import ideal_psf

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXE = os.environ.get("SPECTRAL_RENDER", os.path.join(ROOT, "build", "spectral_render"))


@pytest.mark.skipif(not os.path.exists(EXE), reason="spectral_render not built")
def test_render_then_apply_optics(tmp_path):
    scene = {
        "render": {"spp": 16, "max_depth": 2, "bands": {"min": 400, "max": 1000, "step": 20}},
        "camera": {"position": [0, 1, 4], "look_at": [0, 0.5, 0], "fov_deg": 40, "resolution": [64, 48]},
        "instances": [
            {"primitive": "quad", "size": [20, 20], "seg_id": 1,
             "material": {"reflectance": "approx_asphalt_dry", "specular": 0.0}},
            {"primitive": "sphere", "radius": 0.5, "translate": [0, 0.5, 0], "seg_id": 2,
             "material": {"reflectance": "colorchecker_babel:red"}},
        ],
        "lights": [{"type": "sky", "model": "simple", "sky_radiance": {"type": "cie", "name": "D65", "scale": 0.01},
                    "sun": {"elevation_deg": 40, "azimuth_deg": 150, "irradiance": {"type": "cie", "name": "D65",
                                                                                    "scale": 0.01}}}],
        "output": {"npz": str(tmp_path / "r.npz"), "exr": str(tmp_path / "r.exr"), "metadata": {"frame": 7}},
    }
    (tmp_path / "s.json").write_text(json.dumps(scene))
    subprocess.run([EXE, str(tmp_path / "s.json"), "--quiet"], check=True)
    z = np.load(tmp_path / "r.npz")
    assert z["radiance"].shape == (48, 64, 31)
    assert z["wavelengths"][0] == 400 and z["wavelengths"][-1] == 1000
    meta = json.loads(bytes(z["metadata"]).decode())
    assert meta["frame"] == 7 and meta["camera"]["width"] == 64
    assert set(np.unique(z["seg_id"])) <= {1, 2, 0xFFFFFFFF}
    assert np.isfinite(z["radiance"]).all() and z["radiance"].max() > 0

    wl = np.array([400.0, 700.0, 1000.0])
    psf = np.stack([ideal_psf(w, 2.0, 6.0, 1.0, 33) for w in wl])[:, None]
    pkg = OpticsPackage(wl, [(0.0, 0.0)], psf, 1.0, 6.0, 2.0)
    apply_to_npz(str(tmp_path / "r.npz"), pkg, str(tmp_path / "o.npz"), irradiance=True)
    o = np.load(tmp_path / "o.npz")
    assert o["radiance"].shape == z["radiance"].shape
    ratio = o["radiance"].sum() / z["radiance"].sum()
    assert 0.5 * np.pi / 16 < ratio < 1.05 * np.pi / 16  # camera equation (+cos^4 fall-off)
    assert json.loads(bytes(o["metadata"]).decode())["optics"]["irradiance"] is True
