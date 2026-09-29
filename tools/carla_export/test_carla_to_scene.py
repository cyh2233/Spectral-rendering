"""Offline tests of the CARLA converter (no CARLA installation needed)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carla_to_scene as c  # noqa: E402


def apply(m, p):
    return [sum(m[i][k] * p[k] for k in range(3)) + m[i][3] for i in range(3)]


def test_axis_mapping_is_a_proper_rotation_of_frames():
    # UE forward (+X) -> renderer -Z, UE right (+Y) -> +X, UE up (+Z) -> +Y.
    ident = c.transform_to_matrix({"x": 0, "y": 0, "z": 0}, {"pitch": 0, "yaw": 0, "roll": 0})
    m = c.ue_matrix_to_renderer(ident)
    assert m == [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    tf = c.transform_to_matrix({"x": 10, "y": 2, "z": 1}, {"pitch": 0, "yaw": 90, "roll": 0})
    m = c.ue_matrix_to_renderer(tf)
    assert apply(m, [0, 0, 0]) == [2, 1, -10]
    # A camera yawed +90 deg (looking along UE +Y = renderer +X): its local forward (-Z) maps to +X.
    fwd = [m[i][2] * -1 for i in range(3)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(fwd, [1, 0, 0]))
    # Rotation part stays orthonormal with det +1 in the renderer frame.
    r = [row[:3] for row in m]
    det = (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
           + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))
    assert abs(det - 1) < 1e-12


def test_sun_direction_zenith_and_horizon():
    assert all(abs(a - b) < 1e-12 for a, b in zip(c.sun_direction(90, 0), [0, 1, 0]))
    d = c.sun_direction(0, 0)  # horizon along UE +X = renderer -Z
    assert abs(d[2] + 1) < 1e-12


def test_build_scene_from_snapshot(tmp_path):
    ident = c.transform_to_matrix({"x": 0, "y": 0, "z": 1.5}, {"pitch": 0, "yaw": 0, "roll": 0})
    car = c.transform_to_matrix({"x": 20, "y": 0, "z": 0}, {"pitch": 0, "yaw": 180, "roll": 0})
    snap = {"map": "Carla/Maps/Town10HD_Opt", "frame": 5, "weather": {"sun_altitude_angle": 30, "sun_azimuth_angle": 0},
            "camera": {"matrix": ident, "fov": 90, "width": 640, "height": 360},
            "actors": [{"id": 3, "type_id": "vehicle.tesla.model3", "matrix": car, "semantic_tags": [14],
                        "lights": {"low_beam": True}, "bbox": {"extent": [2.3, 1.0, 0.7], "location": [0, 0, 0.7]}},
                       {"id": 4, "type_id": "static.prop.unknown", "matrix": car}]}
    mapping = {"town": {"Town10HD_Opt": {"path": "town.glb"}},
               "blueprints": {"vehicle.tesla.model3": {"path": "tesla.glb", "seg_id": 14}}}

    class A:
        out = str(tmp_path / "f.json")
        headlights = True
        sun_direction = None

    s = c.build_scene(snap, mapping, A)
    assert [i["asset"] for i in s["instances"]] == ["town", "vehicle_tesla_model3"]
    assert s["instances"][1]["seg_id"] == 14
    assert apply(s["instances"][1]["transform"], [0, 0, 0]) == [0, 0, -20]
    assert len([l for l in s["lights"] if l["type"] == "spot"]) == 2
    assert s["camera"]["resolution"] == [640, 360]
    sky = s["lights"][-1]
    assert abs(sky["sun"]["direction"][1] - math.sin(math.radians(30))) < 1e-12
