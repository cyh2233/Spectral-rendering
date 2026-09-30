"""Keeps a spectral_renderer.Session in sync with a running CARLA world (one call per tick).

Geometry per actor comes from a glTF asset if the blueprint is listed in the config
("blueprints": {"vehicle.tesla.model3": {"path": ".../tesla.glb"}}), otherwise from a proxy box
(bounding box, vehicle colour attribute). Static world: glTF town if configured, else the proxy world
(proxy_world.py). Also synchronises the camera, the sun (weather), vehicle headlights and
traffic-light states. Duck-typed on the CARLA API (tests use fakes).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frames import box_instance_matrix, loc, sun_direction, ue_matrix, ue_point  # noqa: E402
from proxy_world import (LABEL_IDS, environment_instances, ground_instance, label_name,  # noqa: E402
                         road_instances, traffic_light_instances)

# carla.VehicleLightState bit flags.
LIGHT_POSITION, LIGHT_LOW_BEAM, LIGHT_HIGH_BEAM, LIGHT_BRAKE = 0x1, 0x2, 0x4, 0x8


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


class CarlaSync:
    def __init__(self, session, world, config: Dict, camera_actor, render_size=(1280, 720)):
        self.s = session
        self.world = world
        self.cfg = config
        self.camera = camera_actor
        self.render_size = render_size
        self.actors: Dict[int, Dict] = {}          # actor id -> {handle, kind, offset/extent}
        self.traffic_lights: Dict[int, Dict] = {}  # tl id -> {state, handles}
        self.registered_assets = set()
        self.last_sun: Optional[np.ndarray] = None
        self.stats = {"spawned": 0, "removed": 0}

    # ------------------------------------------------------------------ static world
    def build_static(self) -> int:
        """Loads the town (glTF) or builds the proxy world. Returns the number of spawned instances."""
        cfg = self.cfg
        town_name = self.world.get_map().name.split("/")[-1]
        town = cfg.get("town", {}).get(town_name)
        n = 0
        if town:
            self.s.add_asset("town", town["path"])
            inst = {"asset": "town", "seg_id": 0}
            for k in ("seg_id_by_material", "seg_id_by_node", "material_overrides"):
                if k in town:
                    inst[k] = town[k]
            if "fix" in town:
                inst["transform"] = town["fix"]
            self.s.spawn(inst)
            return 1
        proxy = cfg.get("proxy", {})
        step = float(proxy.get("waypoint_step", 2.0))
        wps = list(self.world.get_map().generate_waypoints(step))
        for inst in road_instances(wps, step):
            self.s.spawn(inst)
            n += 1
        if proxy.get("ground", True) and wps:
            self.s.spawn(ground_instance(wps))
            n += 1
        if proxy.get("environment_objects", True):
            for inst in environment_instances(self.world.get_environment_objects()):
                self.s.spawn(inst)
                n += 1
        return n

    # ------------------------------------------------------------------ actors
    def _actor_description(self, actor) -> Dict:
        """Instance description (pose-free) and how to pose it: glTF asset or proxy box."""
        tid = actor.type_id
        tags = list(getattr(actor, "semantic_tags", []) or [])
        seg = int(tags[0]) if tags else (LABEL_IDS["Pedestrians"] if tid.startswith("walker.") else LABEL_IDS["Car"])
        bp = self.cfg.get("blueprints", {}).get(tid)
        if bp:
            key = tid.replace(".", "_")
            if key not in self.registered_assets:
                self.s.add_asset(key, bp["path"])
                self.registered_assets.add(key)
            d = {"asset": key, "seg_id": seg}
            if "material_overrides" in bp:
                d["material_overrides"] = bp["material_overrides"]
            return {"desc": d, "kind": "asset"}
        if tid.startswith("walker."):
            material = "proxy_walker"
        else:
            color = actor.attributes.get("color") if hasattr(actor, "attributes") else None
            rgb = [0.5, 0.5, 0.5]
            if color:
                rgb = [_srgb_to_linear(int(c) / 255.0) for c in color.split(",")[:3]]
            material = {"base_color": rgb, "metallic": 0.2, "roughness": 0.35, "name": "proxy_paint"}
        return {"desc": {"primitive": "box", "size": [1, 1, 1], "material": material, "seg_id": seg}, "kind": "box"}

    def _pose(self, actor, kind: str) -> np.ndarray:
        m = actor.get_transform().get_matrix()
        if kind == "asset":
            return ue_matrix(m)
        bb = actor.bounding_box
        return box_instance_matrix(m, loc(bb.extent), loc(bb.location))

    def _ego_id(self) -> Optional[int]:
        parent = getattr(self.camera, "parent", None)
        return parent.id if parent is not None else None

    def _render_actor(self, actor) -> bool:
        """The vehicle carrying the camera is skipped as a proxy box (the camera would sit inside it); as a glTF
        asset it is rendered only with render_ego (a real cabin/windshield mesh)."""
        if actor.id != self._ego_id():
            return True
        return bool(self.cfg.get("render_ego", False)) and actor.type_id in self.cfg.get("blueprints", {})

    def update_actors(self, actors: List) -> None:
        seen = set()
        for a in actors:
            tid = a.type_id
            if not (tid.startswith("vehicle.") or tid.startswith("walker.")) or not self._render_actor(a):
                continue
            seen.add(a.id)
            rec = self.actors.get(a.id)
            if rec is None:
                info = self._actor_description(a)
                d = dict(info["desc"], transform=self._pose(a, info["kind"]).tolist())
                rec = {"handle": self.s.spawn(d), "kind": info["kind"]}
                self.actors[a.id] = rec
                self.stats["spawned"] += 1
            else:
                self.s.set_transform(rec["handle"], self._pose(a, rec["kind"]))
        for aid in [k for k in self.actors if k not in seen]:
            self.s.remove(self.actors.pop(aid)["handle"])
            self.stats["removed"] += 1

    def update_traffic_lights(self, actors: List) -> None:
        for a in actors:
            if not a.type_id.startswith("traffic.traffic_light"):
                continue
            state = label_name(a.get_state())
            rec = self.traffic_lights.get(a.id)
            if rec and rec["state"] == state:
                continue
            if rec:
                for h in rec["handles"]:
                    self.s.remove(h)
            handles = [self.s.spawn(d) for d in traffic_light_instances(a, state)]
            self.traffic_lights[a.id] = {"state": state, "handles": handles}

    def headlights(self, actors: List) -> List[Dict]:
        hl = self.cfg.get("headlight", {})
        lights = []
        for a in actors:
            if not a.type_id.startswith("vehicle.") or not hasattr(a, "get_light_state"):
                continue
            state = int(a.get_light_state())
            high, low = bool(state & LIGHT_HIGH_BEAM), bool(state & LIGHT_LOW_BEAM)
            if not (high or low):
                continue
            m = np.asarray(a.get_transform().get_matrix(), dtype=np.float64)
            bb = a.bounding_box
            ex, ey, ez = loc(bb.extent)
            lx, ly, lz = loc(bb.location)
            fwd_ue = m[:3, 0] - np.array([0.0, 0.0, hl.get("down_tilt", 0.05)])
            for side in (-1.0, 1.0):
                # Slightly in front of the bounding box so the lamp is not occluded by the proxy body.
                p_ue = m @ np.array([lx + ex + hl.get("offset", 0.05), ly + side * 0.75 * ey, lz - 0.2 * ez, 1.0])
                p = ue_point(p_ue[:3])
                d = ue_point(fwd_ue)
                d = d / np.linalg.norm(d)
                emission = hl.get("emission", {"type": "library", "name": "cie_led_b5",
                                               "photometric": 20000.0 if high else 8000.0})
                lights.append({"type": "spot", "position": p.tolist(), "direction": d.tolist(),
                               "cone_inner_deg": hl.get("cone_inner_deg", 12.0),
                               "cone_outer_deg": hl.get("cone_outer_deg", 30.0), "emission": emission})
        return lights

    # ------------------------------------------------------------------ camera / sun
    def update_camera(self) -> None:
        cam = self.camera
        m = ue_matrix(cam.get_transform().get_matrix())
        fov = float(cam.attributes.get("fov", 90.0))
        w, h = self.render_size
        self.s.set_camera(m, fov, w, h)

    def update_sun(self, min_change_deg: float = 0.2) -> bool:
        w = self.world.get_weather()
        d = sun_direction(float(w.sun_altitude_angle), float(w.sun_azimuth_angle))
        if self.last_sun is not None and math.degrees(math.acos(np.clip(d @ self.last_sun, -1, 1))) < min_change_deg:
            return False
        self.s.set_sun(d)
        self.last_sun = d
        return True

    # ------------------------------------------------------------------ per tick
    def update(self) -> None:
        actors = list(self.world.get_actors())
        self.update_actors(actors)
        self.update_traffic_lights(actors)
        if self.cfg.get("headlights", True):
            self.s.set_dynamic_lights(self.headlights(actors))
        self.update_sun()
        self.update_camera()


def base_scene(config: Dict) -> Dict:
    """Base scene for the live session: proxy materials, render settings, sky."""
    from proxy_world import proxy_materials
    r = config.get("render", {})
    mats = proxy_materials()
    mats.update(config.get("materials", {}))
    sky = config.get("sky", {"type": "sky", "model": "simple",
                              "sky_radiance": {"type": "cie", "name": "D65", "scale": 0.0005},
                              "sun": {"elevation_deg": 45, "azimuth_deg": 0,
                                      "irradiance": {"type": "cie", "name": "D65", "photometric": 100000.0}}})
    sky = dict(sky, type="sky")
    return {
        "camera": {"resolution": list(config.get("render_resolution", [1280, 720])), "fov_deg": 90},
        "materials": {"definitions": mats, "use_default_overrides": True},
        "instances": [],
        "lights": [sky],
        "render": {"spp_per_pass": int(r.get("spp_per_pass", 4)), "max_depth": int(r.get("max_depth", 5)),
                   "bands": r.get("bands", {"min": 380, "max": 1000, "step": 10}), "seed": int(r.get("seed", 0)),
                   "backend": config.get("backend", "auto")},
    }
