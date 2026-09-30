#!/usr/bin/env python3
"""Live CARLA + spectral renderer viewer.

Left: CARLA RGB camera. Right: spectral render of the same camera (preview of the band cube).

  python tools/carla_live/spectral_live.py --config tools/carla_live/config.example.json

Keys
  M      live / quality mode          V      view: sRGB, band, NIR false colour, depth, segmentation
  [ ]    previous / next band         + -    exposure (EV)
  P      pause CARLA (the render keeps refining)
  S      save the current spectral cube (NPZ + EXR + PNG)
  R      record every frame in quality mode
  Esc    quit

Live mode: CARLA runs in synchronous mode; each tick the render restarts with `spp_live` samples.
Quality mode: CARLA waits until `spp_quality` samples are accumulated (progress shown), then ticks.
Requires: CARLA 0.9.15/0.9.16 server + `carla` Python package, pygame, and the spectral_renderer
module (build with -DSPECTRAL_BUILD_PYTHON=ON; PYTHONPATH=build/python).
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import time
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carla_sync import CarlaSync, base_scene  # noqa: E402

VIEWS = ["srgb", "band", "false_color", "depth", "seg"]


def load_config(path: Optional[str]) -> Dict:
    cfg: Dict = {}
    if path:
        with open(path) as f:
            cfg = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        for group in ("blueprints", "town"):
            for v in cfg.get(group, {}).values():
                if "path" in v and not os.path.isabs(v["path"]):
                    v["path"] = os.path.join(base, v["path"])
    return cfg


class CarlaRig:
    """Ego vehicle, RGB camera and background traffic in synchronous mode."""

    def __init__(self, carla_mod, client, cfg: Dict):
        self.carla = carla_mod
        self.client = client
        self.cfg = cfg
        self.world = client.get_world()
        town = cfg.get("carla_town")
        if town and not self.world.get_map().name.endswith(town):
            self.world = client.load_world(town)
        self.original = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = float(cfg.get("fixed_delta_seconds", 0.05))
        self.world.apply_settings(s)
        self.tm = client.get_trafficmanager(int(cfg.get("tm_port", 8000)))
        self.tm.set_synchronous_mode(True)
        self.actors: List = []
        bl = self.world.get_blueprint_library()
        spawns = self.world.get_map().get_spawn_points()
        random.seed(int(cfg.get("seed", 0)))
        ego_bp = bl.filter(cfg.get("ego_blueprint", "vehicle.lincoln.mkz_2020"))[0]
        ego_bp.set_attribute("role_name", "hero")
        self.ego = None
        for sp in random.sample(spawns, len(spawns)):
            self.ego = self.world.try_spawn_actor(ego_bp, sp)
            if self.ego:
                break
        if self.ego is None:
            raise RuntimeError("could not spawn the ego vehicle")
        self.actors.append(self.ego)
        self.ego.set_autopilot(True, self.tm.get_port())
        w, h = cfg.get("carla_resolution", [1280, 720])
        cam_bp = bl.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(w))
        cam_bp.set_attribute("image_size_y", str(h))
        cam_bp.set_attribute("fov", str(cfg.get("fov", 90)))
        mount = cfg.get("camera_mount", {"x": 0.8, "y": 0.0, "z": 1.3, "pitch": 0.0})
        tf = carla_mod.Transform(carla_mod.Location(x=mount["x"], y=mount["y"], z=mount["z"]),
                                 carla_mod.Rotation(pitch=mount.get("pitch", 0.0)))
        self.camera = self.world.spawn_actor(cam_bp, tf, attach_to=self.ego)
        self.actors.append(self.camera)
        self.images: "queue.Queue" = queue.Queue()
        self.camera.listen(self.images.put)
        vbps = [b for b in bl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels").as_int()) == 4]
        for sp in random.sample(spawns, min(len(spawns), int(cfg.get("traffic", 20)))):
            v = self.world.try_spawn_actor(random.choice(vbps), sp)
            if v:
                v.set_autopilot(True, self.tm.get_port())
                self.actors.append(v)
        self.world.tick()

    def tick(self):
        frame = self.world.tick()
        while True:
            img = self.images.get(timeout=10.0)
            if img.frame >= frame:
                return img

    def close(self):
        try:
            self.camera.stop()
        except Exception:
            pass
        for a in reversed(self.actors):
            try:
                a.destroy()
            except Exception:
                pass
        self.world.apply_settings(self.original)
        self.tm.set_synchronous_mode(False)


def carla_image_rgb(img) -> np.ndarray:
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    return a[:, :, 2::-1].copy()  # BGRA -> RGB


class Viewer:
    def __init__(self, session, rig, sync, cfg: Dict, headless: bool = False):
        import pygame
        self.pg = pygame
        self.s, self.rig, self.sync, self.cfg = session, rig, sync, cfg
        self.panel = tuple(cfg.get("panel_size", [960, 540]))
        self.view = 0
        self.band_nm = 550.0
        self.ev = 0.0
        self.quality = False
        self.paused = False
        self.record = False
        self.frame = 0
        self.save_dir = cfg.get("save_dir", "carla_live_out")
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        self.screen = pygame.display.set_mode((2 * self.panel[0], self.panel[1] + 28))
        pygame.display.set_caption("CARLA | spectral")
        self.font = pygame.font.SysFont("monospace", 16)
        self.last_carla = np.zeros((self.panel[1], self.panel[0], 3), np.uint8)
        self.fps = 0.0

    def _blit(self, rgb: np.ndarray, x0: int) -> None:
        surf = self.pg.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
        surf = self.pg.transform.smoothscale(surf, self.panel)
        self.screen.blit(surf, (x0, 0))

    def _preview(self) -> np.ndarray:
        mode = VIEWS[self.view]
        return self.s.preview(mode, band_nm=self.band_nm, exposure_ev=self.ev,
                              rgb_nm=tuple(self.cfg.get("false_color_nm", [850, 650, 550])))

    def draw(self, status: str = "") -> None:
        self.screen.fill((0, 0, 0))
        self._blit(self.last_carla, 0)
        self._blit(self._preview(), self.panel[0])
        view = VIEWS[self.view] + (f" {self.band_nm:.0f}nm" if VIEWS[self.view] == "band" else "")
        hud = (f"{'QUALITY' if self.quality else 'LIVE'}{' PAUSED' if self.paused else ''}{' REC' if self.record else ''}"
               f" | {view} | EV {self.ev:+.1f} | spp {self.s.spp} | {self.fps:4.1f} fps | "
               f"objects {self.s.live_objects()} | {self.s.backend} {status}")
        self.screen.blit(self.font.render(hud, True, (230, 230, 230)), (6, self.panel[1] + 5))
        self.pg.display.flip()

    def save(self, tag: str) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        base = os.path.join(self.save_dir, tag)
        self.s.save(npz=base + ".npz", exr=base + ".exr", preview_png=base + "_srgb.png")
        return base

    def handle_events(self) -> bool:
        pg = self.pg
        wl = self.s.wavelengths
        for e in pg.event.get():
            if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                return False
            if e.type != pg.KEYDOWN:
                continue
            if e.key == pg.K_m:
                self.quality = not self.quality
            elif e.key == pg.K_v:
                self.view = (self.view + 1) % len(VIEWS)
            elif e.key == pg.K_RIGHTBRACKET:
                self.band_nm = float(min(wl[-1], self.band_nm + (wl[1] - wl[0])))
            elif e.key == pg.K_LEFTBRACKET:
                self.band_nm = float(max(wl[0], self.band_nm - (wl[1] - wl[0])))
            elif e.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self.ev += 0.5
            elif e.key in (pg.K_MINUS, pg.K_KP_MINUS):
                self.ev -= 0.5
            elif e.key == pg.K_p:
                self.paused = not self.paused
            elif e.key == pg.K_r:
                self.record = not self.record
            elif e.key == pg.K_s:
                print("saved", self.save(f"snapshot_{self.frame:06d}"))
        return True

    def step(self) -> bool:
        """One viewer iteration. Returns False to quit."""
        if not self.handle_events():
            return False
        t0 = time.time()
        if not self.paused:
            img = self.rig.tick()
            self.frame += 1
            self.last_carla = carla_image_rgb(img)
            self.sync.update()
            self.s.reset()
        if self.quality and not self.paused:
            target, chunk = int(self.cfg.get("spp_quality", 256)), int(self.cfg.get("spp_chunk", 16))
            while self.s.spp < target:
                self.s.render(min(chunk, target - self.s.spp))
                self.draw(f"| rendering {self.s.spp}/{target}")
                if not self.handle_events():
                    return False
            if self.record:
                self.save(f"frame_{self.frame:06d}")
        else:
            self.s.render(int(self.cfg.get("spp_live", 4)))
        self.fps = 0.8 * self.fps + 0.2 / max(1e-6, time.time() - t0)
        self.draw()
        return True


def run(carla_mod, client, cfg: Dict, max_frames: int = 0, headless: bool = False, session_cls=None) -> Viewer:
    if session_cls is None:
        from spectral_renderer import Session as session_cls
    rig = CarlaRig(carla_mod, client, cfg)
    viewer = None
    try:
        session = session_cls(base_scene(cfg), backend=cfg.get("backend", "auto"),
                              base_dir=cfg.get("base_dir", os.getcwd()))
        sync = CarlaSync(session, rig.world, cfg, rig.camera, tuple(cfg.get("render_resolution", [1280, 720])))
        t = time.time()
        n = sync.build_static()
        print(f"[live] static world: {n} instances in {time.time() - t:.1f}s ({session.backend} backend)")
        viewer = Viewer(session, rig, sync, cfg, headless=headless)
        frames = 0
        while viewer.step():
            frames += 1
            if max_frames and frames >= max_frames:
                break
    finally:
        rig.close()
    return viewer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON config (see config.example.json)")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--backend", default=None, help="cuda | cpu | auto")
    ap.add_argument("--frames", type=int, default=0, help="stop after N frames (0 = run until Esc)")
    ap.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.backend:
        cfg["backend"] = a.backend
    import carla  # noqa: WPS433
    client = carla.Client(a.host or cfg.get("host", "localhost"), a.port or int(cfg.get("port", 2000)))
    client.set_timeout(float(cfg.get("timeout", 20.0)))
    run(carla, client, cfg, a.frames, a.headless)
    return 0


if __name__ == "__main__":
    sys.exit(main())
