"""Python bindings of the live session (build with -DSPECTRAL_BUILD_PYTHON=ON; PYTHONPATH=build/python)."""
import json
import os

import numpy as np
import pytest

sr = pytest.importorskip("spectral_renderer")

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

SCENE = {
    "render": {"spp": 4, "seed": 1, "bands": {"min": 400, "max": 1000, "step": 20}},
    "camera": {"position": [0, 1.5, 6], "look_at": [0, 0.5, 0], "fov_deg": 50, "resolution": [64, 40]},
    "instances": [{"primitive": "quad", "size": [30, 30], "seg_id": 1, "material": {"reflectance": 0.4}}],
    "lights": [{"type": "sky", "model": "simple", "sky_radiance": 0.05,
                "sun": {"elevation_deg": 40, "azimuth_deg": 30, "irradiance": 5.0}}],
}


def translation(x, y, z):
    m = np.eye(4)[:3]
    m[:, 3] = [x, y, z]
    return m


def test_spawn_move_render_preview(tmp_path):
    s = sr.Session(SCENE, backend="cpu", threads=2)
    assert s.size == (64, 40)
    assert s.wavelengths[0] == 400 and len(s.wavelengths) == 31
    car = s.spawn({"primitive": "box", "size": [1.8, 1.4, 4.4], "seg_id": 14,
                   "material": {"reflectance": "approx_car_paint_red"}})
    s.set_transform(car, translation(0, 0.7, 0))
    assert s.render(4) == 4
    rgb = s.preview("srgb")
    assert rgb.shape == (40, 64, 3) and rgb.dtype == np.uint8
    img = s.image()
    assert img["radiance"].shape == (40, 64, 31)
    assert (img["seg_id"] == 14).sum() > 20
    red = img["radiance"][img["seg_id"] == 14].mean(axis=0)
    assert red[s.wavelengths.tolist().index(660)] > 3 * red[s.wavelengths.tolist().index(460)]
    # Moving the car restarts accumulation.
    s.set_transform(car, translation(3, 0.7, 0))
    assert s.render(2) == 2
    s.remove(car)
    s.render(1)
    assert (s.image()["seg_id"] == 14).sum() == 0
    s.save(npz=str(tmp_path / "f.npz"), preview_png=str(tmp_path / "f.png"))
    z = np.load(tmp_path / "f.npz")
    assert json.loads(bytes(z["metadata"]).decode())["camera"]["width"] == 64


def test_camera_lights_and_modes():
    s = sr.Session(SCENE, backend="cpu", threads=1)
    s.set_camera(np.array([[1, 0, 0, 0], [0, 1, 0, 1.2], [0, 0, 1, 5]], float), 70.0, 32, 20)
    s.set_dynamic_lights([{"type": "spot", "position": [0, 1, 2], "direction": [0, -0.2, -1],
                           "cone_inner_deg": 10, "cone_outer_deg": 25,
                           "emission": {"type": "library", "name": "cie_led_b5", "photometric": 5000}}])
    s.set_sun([0.3, 0.2, 0.93])
    s.render(2)
    for mode in ("srgb", "band", "false_color", "depth", "seg"):
        assert s.preview(mode).shape == (20, 32, 3)
    with pytest.raises(Exception):
        s.preview("bogus")
