#!/usr/bin/env python3
"""CARLA -> spectral renderer scene JSON (offline workflow).

The static town and the actor blueprints are exported ONCE to glTF (see docs/carla_export.md).
For every frame this script reads the actor transforms, the camera sensor and the weather from a
running CARLA server (or a recorded JSON snapshot) and writes a scene JSON that references those
glTF assets. The renderer itself never depends on CARLA.

Coordinate frames
  CARLA / Unreal: left-handed, X forward, Y right, Z up, metres.
  Renderer / glTF: right-handed, X right, Y up, Z backward (-Z forward), metres.
  Mapping S: (x, y, z)_UE -> (y, z, -x). Transforms convert as  M = S * M_UE * S^T.
  Assets exported to glTF by Unreal's glTF exporter already use glTF axes; if your exporter
  produced another convention, correct it with the per-asset "fix" matrix in the mapping file.

Usage
  python carla_to_scene.py --mapping asset_mapping.json --out frame_000123.json \
         [--host localhost --port 2000 --camera-role hero_camera] [--snapshot snapshot.json]
  python carla_to_scene.py --dump-snapshot snapshot.json ...   # save CARLA state for offline use
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Dict, List

S = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]

# CARLA CityObjectLabel (0.9.14+) -> segmentation ids used in the output (identity by default).
CITYSCAPES_NAMES = {0: "unlabeled", 1: "road", 2: "sidewalk", 3: "building", 4: "wall", 5: "fence", 6: "pole",
                    7: "traffic_light", 8: "traffic_sign", 9: "vegetation", 10: "terrain", 11: "sky",
                    12: "pedestrian", 13: "rider", 14: "car", 15: "truck", 16: "bus", 17: "train",
                    18: "motorcycle", 19: "bicycle", 20: "static", 21: "dynamic", 22: "other", 23: "water",
                    24: "road_line", 25: "ground", 26: "bridge", 27: "rail_track", 28: "guard_rail"}


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a):
    return [list(r) for r in zip(*a)]


def ue_matrix_to_renderer(m4) -> List[List[float]]:
    """4x4 (row-major, UE convention, metres) -> 3x4 row-major renderer matrix."""
    r = [row[:3] for row in m4[:3]]
    t = [m4[i][3] for i in range(3)]
    rr = matmul(matmul(S, r), transpose(S))
    tt = [sum(S[i][k] * t[k] for k in range(3)) for i in range(3)]
    return [rr[i] + [tt[i]] for i in range(3)]


def transform_to_matrix(loc, rot) -> List[List[float]]:
    """CARLA Location (m) + Rotation (deg: pitch, yaw, roll) -> 4x4 UE matrix (as carla.Transform.get_matrix)."""
    cy, sy = math.cos(math.radians(rot["yaw"])), math.sin(math.radians(rot["yaw"]))
    cr, sr = math.cos(math.radians(rot["roll"])), math.sin(math.radians(rot["roll"]))
    cp, sp = math.cos(math.radians(rot["pitch"])), math.sin(math.radians(rot["pitch"]))
    return [[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, loc["x"]],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, loc["y"]],
            [sp, -cp * sr, cp * cr, loc["z"]],
            [0.0, 0.0, 0.0, 1.0]]


def sun_direction(altitude_deg: float, azimuth_deg: float):
    """CARLA weather sun angles -> unit vector towards the sun in renderer coordinates.
    CARLA: altitude above the horizon, azimuth measured in the UE XY plane from +X towards +Y.
    (Verify against your CARLA version; override with --sun-direction if needed.)"""
    alt, az = math.radians(altitude_deg), math.radians(azimuth_deg)
    ue = [math.cos(alt) * math.cos(az), math.cos(alt) * math.sin(az), math.sin(alt)]
    return [sum(S[i][k] * ue[k] for k in range(3)) for i in range(3)]


# ---------------------------------------------------------------------------- CARLA access
def snapshot_from_carla(host: str, port: int, camera_role: str) -> Dict:
    import carla  # noqa: WPS433 (only needed online)

    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    snap = {"map": world.get_map().name, "frame": world.get_snapshot().frame, "actors": [], "camera": None}
    w = world.get_weather()
    snap["weather"] = {"sun_altitude_angle": w.sun_altitude_angle, "sun_azimuth_angle": w.sun_azimuth_angle,
                       "cloudiness": w.cloudiness, "fog_density": w.fog_density, "wetness": w.wetness}
    for a in world.get_actors():
        tf = a.get_transform()
        rec = {"id": a.id, "type_id": a.type_id, "role_name": a.attributes.get("role_name", ""),
               "matrix": [list(r) for r in tf.get_matrix()],
               "semantic_tags": list(getattr(a, "semantic_tags", []) or [])}
        if a.type_id.startswith("vehicle."):
            try:
                ls = a.get_light_state()
                rec["lights"] = {"low_beam": bool(ls & carla.VehicleLightState.LowBeam),
                                 "high_beam": bool(ls & carla.VehicleLightState.HighBeam),
                                 "brake": bool(ls & carla.VehicleLightState.Brake),
                                 "position": bool(ls & carla.VehicleLightState.Position)}
            except RuntimeError:
                pass
            bb = a.bounding_box
            rec["bbox"] = {"extent": [bb.extent.x, bb.extent.y, bb.extent.z],
                           "location": [bb.location.x, bb.location.y, bb.location.z]}
        if a.type_id.startswith("traffic.traffic_light"):
            rec["state"] = str(a.get_state())
        if a.type_id.startswith("sensor.camera") and (not camera_role or rec["role_name"] == camera_role):
            snap["camera"] = {"matrix": rec["matrix"], "fov": float(a.attributes["fov"]),
                              "width": int(a.attributes["image_size_x"]), "height": int(a.attributes["image_size_y"])}
        snap["actors"].append(rec)
    if snap["camera"] is None:
        raise RuntimeError(f"no camera sensor with role_name '{camera_role}' found")
    return snap


# ---------------------------------------------------------------------------- scene generation
def headlight_lights(actor, mapping) -> List[Dict]:
    """Two spot lights at the front corners of the bounding box for vehicles with low/high beams on."""
    lights = actor.get("lights", {})
    if not (lights.get("low_beam") or lights.get("high_beam")) or "bbox" not in actor:
        return []
    ex, ey, ez = actor["bbox"]["extent"]
    lx, ly, lz = actor["bbox"]["location"]
    m = actor["matrix"]
    out = []
    hl = mapping.get("headlight", {})
    for side in (-1, 1):
        local = [lx + ex, ly + side * 0.75 * ey, lz - 0.2 * ez, 1.0]
        p_ue = [sum(m[i][k] * local[k] for k in range(4)) for i in range(3)]
        fwd = [m[0][0], m[1][0], m[2][0] - hl.get("down_tilt", 0.03)]
        p = [sum(S[i][k] * p_ue[k] for k in range(3)) for i in range(3)]
        d = [sum(S[i][k] * fwd[k] for k in range(3)) for i in range(3)]
        n = math.sqrt(sum(v * v for v in d))
        out.append({"type": "spot", "name": f"headlight_{actor['id']}_{'L' if side < 0 else 'R'}", "position": p,
                    "direction": [v / n for v in d], "cone_inner_deg": hl.get("cone_inner_deg", 12.0),
                    "cone_outer_deg": hl.get("cone_outer_deg", 30.0),
                    "emission": hl.get("emission", {"type": "library", "name": "cie_led_b5",
                                                    "photometric": 20000.0 if lights.get("high_beam") else 8000.0})})
    return out


def build_scene(snap: Dict, mapping: Dict, args) -> Dict:
    cam = snap["camera"]
    cam_m = ue_matrix_to_renderer(cam["matrix"])
    scene = {
        "version": 2,
        "assets": {},
        "camera": {"type": "pinhole", "matrix": cam_m, "fov_deg": cam["fov"], "fov_axis": "horizontal",
                   "resolution": [cam["width"], cam["height"]]},
        "instances": [],
        "materials": {"use_default_overrides": True, "overrides": mapping.get("material_overrides", [])},
        "lights": [],
        "render": mapping.get("render", {"spp": 256, "max_depth": 8, "bands": {"min": 380, "max": 1000, "step": 5}}),
        "output": {"exr": args.out.replace(".json", ".exr"), "npz": args.out.replace(".json", ".npz"),
                   "metadata": {"source": "carla", "map": snap.get("map"), "frame": snap.get("frame")}},
    }
    town = mapping.get("town", {}).get(snap.get("map", "").split("/")[-1])
    if town:
        scene["assets"]["town"] = {"path": town["path"]}
        inst = {"asset": "town", "seg_id": 0, "seg_id_by_material": town.get("seg_id_by_material", {}),
                "seg_id_by_node": town.get("seg_id_by_node", {})}
        if "fix" in town:
            inst["transform"] = town["fix"]
        scene["instances"].append(inst)
    else:
        print(f"warning: no town asset for map {snap.get('map')}", file=sys.stderr)
    blueprints = mapping.get("blueprints", {})
    for a in snap["actors"]:
        bp = blueprints.get(a["type_id"])
        if bp is None:
            continue
        key = a["type_id"].replace(".", "_")
        scene["assets"].setdefault(key, {"path": bp["path"]})
        tags = a.get("semantic_tags") or [bp.get("seg_id", 0)]
        inst = {"asset": key, "transform": ue_matrix_to_renderer(a["matrix"]), "seg_id": int(tags[0]),
                "name": f"{a['type_id']}#{a['id']}"}
        if "material_overrides" in bp:
            inst["material_overrides"] = bp["material_overrides"]
        scene["instances"].append(inst)
        if args.headlights:
            scene["lights"] += headlight_lights(a, mapping)
    w = snap.get("weather", {})
    sky = dict(mapping.get("sky", {"type": "sky", "model": "simple",
                                   "sky_radiance": {"type": "cie", "name": "D65", "scale": 0.0005}}))
    sky.setdefault("type", "sky")
    sun = dict(sky.get("sun", {}))
    sun["direction"] = args.sun_direction or sun_direction(w.get("sun_altitude_angle", 45.0),
                                                           w.get("sun_azimuth_angle", 0.0))
    if sky.get("model") == "simple":
        sun.setdefault("irradiance", {"type": "cie", "name": "D65", "photometric": 100000.0})
    sky["sun"] = sun
    scene["lights"].append(sky)
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapping", required=True, help="asset mapping JSON (see asset_mapping.example.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot", help="read a saved snapshot instead of connecting to CARLA")
    ap.add_argument("--dump-snapshot", help="also save the CARLA snapshot to this file")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--camera-role", default="", help="role_name of the camera sensor (default: first camera)")
    ap.add_argument("--headlights", action="store_true", help="add spot lights for vehicles with beams on")
    ap.add_argument("--sun-direction", type=float, nargs=3, help="override sun direction (renderer frame)")
    args = ap.parse_args(argv)
    if args.snapshot:
        with open(args.snapshot) as f:
            snap = json.load(f)
    else:
        snap = snapshot_from_carla(args.host, args.port, args.camera_role)
    if args.dump_snapshot:
        with open(args.dump_snapshot, "w") as f:
            json.dump(snap, f, indent=1)
    with open(args.mapping) as f:
        mapping = json.load(f)
    scene = build_scene(snap, mapping, args)
    with open(args.out, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"wrote {args.out}: {len(scene['instances'])} instances, {len(scene['lights'])} lights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
