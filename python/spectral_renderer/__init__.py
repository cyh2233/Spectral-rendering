"""Python front end of the spectral renderer's live session.

    from spectral_renderer import Session
    s = Session("scene.json", backend="cuda")
    car = s.spawn({"primitive": "box", "size": [1.8, 1.5, 4.5], "material": {"reflectance": "approx_car_paint_red"}})
    s.set_transform(car, matrix_3x4)
    s.set_camera(cam_to_world_3x4, fov_x_deg=90, width=1280, height=720)
    s.render(4)
    rgb = s.preview("srgb")          # (H, W, 3) uint8
    cube = s.image()["radiance"]     # (H, W, bands) float32, W/(m^2 sr nm)
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

import numpy as np

from ._spectral import Session as _Session, __version__  # noqa: F401


def _js(x: Any) -> str:
    return x if isinstance(x, str) else json.dumps(x)


class Session:
    """Thin wrapper accepting dicts / numpy arrays; see include/spectral/session.h."""

    def __init__(self, scene: Any = None, backend: str = "auto", threads: int = 0, base_dir: Optional[str] = None):
        self._s = _Session(backend, threads)
        if scene is None:
            return
        if isinstance(scene, str) and os.path.exists(scene):
            self._s.load_file(scene)
        else:
            self._s.load_json(_js(scene), base_dir or os.getcwd())

    # objects
    def add_asset(self, key: str, path: str) -> None:
        self._s.add_asset(key, path)

    def spawn(self, instance: Any) -> int:
        return self._s.spawn(_js(instance))

    def set_transform(self, handle: int, matrix) -> None:
        self._s.set_transform(handle, np.asarray(matrix, dtype=np.float64))

    def set_seg_id(self, handle: int, seg_id: int) -> None:
        self._s.set_seg_id(handle, int(seg_id))

    def remove(self, handle: int) -> None:
        self._s.remove(handle)

    def live_objects(self) -> int:
        return self._s.live_objects()

    # camera / lights
    def set_camera(self, cam_to_world, fov_x_deg: float, width: int, height: int) -> None:
        self._s.set_camera(np.asarray(cam_to_world, dtype=np.float64), float(fov_x_deg), int(width), int(height))

    def set_dynamic_lights(self, lights: Iterable[dict]) -> None:
        self._s.set_dynamic_lights(_js(list(lights)))

    def set_sun(self, direction) -> None:
        d = np.asarray(direction, dtype=np.float64)
        self._s.set_sun(float(d[0]), float(d[1]), float(d[2]))

    # rendering
    def reset(self) -> None:
        self._s.reset()

    def render(self, spp: int) -> int:
        return self._s.render(int(spp))

    def preview(self, mode: str = "srgb", band_nm: float = 550.0, rgb_nm=(850.0, 650.0, 550.0),
                auto_exposure: bool = True, exposure_ev: float = 0.0, depth_max: float = 100.0) -> np.ndarray:
        return self._s.preview(mode, float(band_nm), [float(v) for v in rgb_nm], auto_exposure, float(exposure_ev),
                               float(depth_max))

    def image(self) -> dict:
        return self._s.image()

    def save(self, npz: str = "", exr: str = "", preview_png: str = "") -> None:
        self._s.save(npz, exr, preview_png)

    @property
    def spp(self) -> int:
        return self._s.spp

    @property
    def backend(self) -> str:
        return self._s.backend

    @property
    def wavelengths(self) -> np.ndarray:
        return self._s.wavelengths

    @property
    def size(self):
        return self._s.width, self._s.height
