"""Proxy geometry from a packaged CARLA (no Unreal editor / asset export needed).

  * roads: triangle strips along map waypoints (lane width), per-lane-type materials
  * lane markings: thin strips along lane edges (solid / broken, white / yellow)
  * static objects: oriented boxes from world.get_environment_objects(), material by semantic class
  * ground plane under the map

Everything is emitted as renderer instance descriptions (dicts). Functions are duck-typed on the
CARLA API so they can be tested without CARLA.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np

from frames import box_instance_matrix, loc, rot, ue_point, ue_transform_matrix

# CARLA CityObjectLabel names -> renderer material names (defined in the live base scene).
DEFAULT_CLASS_MATERIALS = {
    "Buildings": "proxy_building", "Walls": "proxy_building", "Fences": "proxy_metal", "Poles": "proxy_metal",
    "TrafficSigns": "proxy_sign", "Vegetation": "proxy_vegetation", "Terrain": "proxy_vegetation",
    "Sidewalks": "proxy_concrete", "Static": "proxy_concrete", "Other": "proxy_concrete", "Bridge": "proxy_concrete",
    "GuardRail": "proxy_metal", "Water": "proxy_water", "RailTrack": "proxy_metal", "Ground": "proxy_concrete",
}
# Classes represented otherwise (roads from waypoints, dynamic actors, sky) are skipped.
SKIP_CLASSES = {"Roads", "RoadLines", "Sky", "Car", "Truck", "Bus", "Motorcycle", "Bicycle", "Pedestrians", "Rider",
                "Train", "TrafficLight", "Dynamic", "None", "Any"}

# CARLA CityObjectLabel ids (0.9.14+), used as segmentation ids.
LABEL_IDS = {"None": 0, "Roads": 1, "Sidewalks": 2, "Buildings": 3, "Walls": 4, "Fences": 5, "Poles": 6,
             "TrafficLight": 7, "TrafficSigns": 8, "Vegetation": 9, "Terrain": 10, "Sky": 11, "Pedestrians": 12,
             "Rider": 13, "Car": 14, "Truck": 15, "Bus": 16, "Train": 17, "Motorcycle": 18, "Bicycle": 19,
             "Static": 20, "Dynamic": 21, "Other": 22, "Water": 23, "RoadLines": 24, "Ground": 25, "Bridge": 26,
             "RailTrack": 27, "GuardRail": 28}


def label_name(label) -> str:
    """carla.CityObjectLabel (enum) or plain string/int -> name."""
    if isinstance(label, str):
        return label
    name = getattr(label, "name", None)
    if name:
        return str(name)
    s = str(label)
    if s.isdigit():
        inv = {v: k for k, v in LABEL_IDS.items()}
        return inv.get(int(s), "None")
    return s.split(".")[-1]


class _Mesh:
    def __init__(self):
        self.p: List[float] = []
        self.i: List[int] = []

    def quad(self, a, b, c, d, up=(0.0, 1.0, 0.0)):
        """Quad a-b-c-d (renderer coordinates); triangles oriented so the normal faces `up`."""
        base = len(self.p) // 3
        for v in (a, b, c, d):
            self.p.extend(float(x) for x in v)
        n = np.cross(np.subtract(b, a), np.subtract(c, a))
        if np.dot(n, up) >= 0:
            self.i.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        else:
            self.i.extend([base, base + 2, base + 1, base, base + 3, base + 2])

    def instance(self, material: str, seg_id: int, name: str) -> Dict:
        return {"primitive": "mesh", "name": name, "positions": self.p, "indices": self.i, "material": material,
                "seg_id": seg_id}


def _edges(wp, lift=0.0):
    t = wp.transform
    c = np.array(loc(t.location))
    r = np.array(loc(t.get_right_vector()))
    h = 0.5 * float(wp.lane_width)
    left, right = c - r * h, c + r * h
    left[2] += lift
    right[2] += lift
    return left, right


def road_instances(waypoints: Iterable, step: float, lane_material: Dict[str, str] | None = None,
                   marking_width: float = 0.15, dash: float = 3.0) -> List[Dict]:
    """Road surface + lane markings from waypoints (as from carla.Map.generate_waypoints(step))."""
    lane_material = lane_material or {"Driving": "proxy_asphalt", "Parking": "proxy_asphalt",
                                      "Shoulder": "proxy_asphalt", "Sidewalk": "proxy_concrete",
                                      "Biking": "proxy_asphalt"}
    surfaces: Dict[str, _Mesh] = {}
    white, yellow = _Mesh(), _Mesh()
    for wp in waypoints:
        lane = label_name(getattr(wp, "lane_type", "Driving"))
        mat = lane_material.get(lane)
        if mat is None:
            continue
        for nxt in wp.next(step):
            if nxt.road_id != wp.road_id or nxt.lane_id != wp.lane_id:
                continue
            l0, r0 = _edges(wp)
            l1, r1 = _edges(nxt)
            surfaces.setdefault(mat, _Mesh()).quad(ue_point(l0), ue_point(r0), ue_point(r1), ue_point(l1))
            for side, (e0, e1) in (("right", (r0, r1)), ("left", (l0, l1))):
                mk = getattr(wp, f"{side}_lane_marking", None)
                if mk is None:
                    continue
                kind = label_name(mk.type)
                if kind in ("NONE", "Other", "Grass", "Curb"):
                    continue
                if kind.startswith("Broken") and int(float(getattr(wp, "s", 0.0)) / dash) % 2:
                    continue
                d = np.array(loc(wp.transform.get_right_vector())) * (0.5 * marking_width)
                lift = np.array([0.0, 0.0, 0.01])
                target = yellow if label_name(mk.color) == "Yellow" else white
                target.quad(ue_point(e0 - d + lift), ue_point(e0 + d + lift), ue_point(e1 + d + lift),
                            ue_point(e1 - d + lift))
    out = []
    for mat, m in surfaces.items():
        seg = LABEL_IDS["Sidewalks"] if mat == "proxy_concrete" else LABEL_IDS["Roads"]
        out.append(m.instance(mat, seg, f"road_{mat}"))
    if white.i:
        out.append(white.instance("proxy_marking_white", LABEL_IDS["RoadLines"], "markings_white"))
    if yellow.i:
        out.append(yellow.instance("proxy_marking_yellow", LABEL_IDS["RoadLines"], "markings_yellow"))
    return out


def ground_instance(waypoints: Iterable, margin: float = 200.0, material: str = "proxy_vegetation") -> Dict:
    pts = np.array([loc(w.transform.location) for w in waypoints])
    lo, hi = pts.min(axis=0) - margin, pts.max(axis=0) + margin
    z = float(pts[:, 2].min()) - 0.05
    m = _Mesh()
    m.quad(ue_point((lo[0], lo[1], z)), ue_point((hi[0], lo[1], z)), ue_point((hi[0], hi[1], z)),
           ue_point((lo[0], hi[1], z)))
    return m.instance(material, LABEL_IDS["Terrain"], "ground")


def environment_instances(env_objects: Iterable, class_materials: Dict[str, str] | None = None,
                          max_extent: float = 400.0) -> List[Dict]:
    """Oriented unit boxes for world.get_environment_objects() (world-space bounding boxes)."""
    class_materials = class_materials or DEFAULT_CLASS_MATERIALS
    out = []
    for obj in env_objects:
        name = label_name(obj.type)
        if name in SKIP_CLASSES or name not in class_materials:
            continue
        bb = obj.bounding_box
        ext = loc(bb.extent)
        if max(ext) > max_extent or min(ext) <= 0.0:
            continue  # terrain-sized boxes would swallow the scene
        m = box_instance_matrix(ue_transform_matrix(loc(bb.location), rot(bb.rotation)), ext)
        out.append({"primitive": "box", "size": [1, 1, 1], "material": class_materials[name],
                    "transform": m.tolist(), "seg_id": LABEL_IDS.get(name, 0),
                    "name": f"{name}_{getattr(obj, 'id', 0)}"})
    return out


def traffic_light_instances(tl, state: str) -> List[Dict]:
    """Emissive boxes on the light heads of a traffic light for the current state (Red/Yellow/Green)."""
    material = {"Red": "proxy_signal_red", "Yellow": "proxy_signal_amber", "Green": "proxy_signal_green"}.get(state)
    out = []
    for k, bb in enumerate(tl.get_light_boxes()):
        m = box_instance_matrix(ue_transform_matrix(loc(bb.location), rot(bb.rotation)), loc(bb.extent))
        out.append({"primitive": "box", "size": [1, 1, 1], "material": material or "proxy_signal_off",
                    "transform": m.tolist(), "seg_id": LABEL_IDS["TrafficLight"], "name": f"tl_{tl.id}_{k}"})
    return out


def proxy_materials() -> Dict[str, Dict]:
    """Material definitions used by the proxy world (merged into the live base scene)."""
    def led(center, width, cd_m2):
        w = [center - 2 * width, center - width, center, center + width, center + 2 * width]
        return {"type": "samples", "wavelengths": w, "values": [0.0, 0.5, 1.0, 0.5, 0.0], "extrapolation": "zero",
                "photometric": cd_m2}
    return {
        "proxy_asphalt": {"reflectance": "approx_asphalt_dry", "roughness": 0.9, "specular": 0.5},
        "proxy_concrete": {"reflectance": "approx_concrete", "roughness": 0.8, "specular": 0.5},
        "proxy_building": {"reflectance": "approx_concrete", "roughness": 0.7},
        "proxy_vegetation": {"reflectance": "approx_vegetation_leaf", "roughness": 0.9, "specular": 0.2},
        "proxy_metal": {"reflectance": 0.55, "metallic": 0.8, "roughness": 0.4},
        "proxy_sign": {"reflectance": "approx_retroreflective_sheeting_white", "roughness": 0.3},
        "proxy_water": {"reflectance": 0.05, "roughness": 0.05},
        "proxy_marking_white": {"reflectance": "approx_lane_marking_white", "roughness": 0.6},
        "proxy_marking_yellow": {"reflectance": "approx_lane_marking_yellow", "roughness": 0.6},
        "proxy_walker": {"reflectance": "approx_skin_light", "roughness": 0.6},
        "proxy_signal_off": {"reflectance": 0.05, "roughness": 0.3},
        "proxy_signal_red": {"reflectance": 0.05, "emission": led(625.0, 10.0, 6000.0), "double_sided": True},
        "proxy_signal_amber": {"reflectance": 0.05, "emission": led(592.0, 8.0, 6000.0), "double_sided": True},
        "proxy_signal_green": {"reflectance": 0.05, "emission": led(505.0, 15.0, 6000.0), "double_sided": True},
    }
