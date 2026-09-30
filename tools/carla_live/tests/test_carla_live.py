"""Live viewer against a fake CARLA world (no CARLA server, CPU backend, headless pygame)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

sr = pytest.importorskip("spectral_renderer")
pytest.importorskip("pygame")

import fake_carla  # noqa: E402
import frames  # noqa: E402
import spectral_live  # noqa: E402
from proxy_world import LABEL_IDS, environment_instances, road_instances  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))


def cfg(tmp_path):
    return {"backend": "cpu", "render_resolution": [96, 54], "carla_resolution": [64, 36], "panel_size": [96, 54],
            "spp_live": 2, "spp_quality": 8, "spp_chunk": 4, "traffic": 0, "fov": 90,
            "render": {"bands": {"min": 400, "max": 1000, "step": 20}, "max_depth": 3},
            "base_dir": ROOT, "save_dir": str(tmp_path / "out")}


def test_frames_mapping():
    m = frames.ue_matrix(frames.ue_transform_matrix((10, 2, 1), (0, 90, 0)))
    assert np.allclose(m[:, 3], [2, 1, -10])
    assert np.allclose(-m[:, 2], [1, 0, 0])  # camera forward (UE yaw 90 = +Y = renderer +X)
    b = frames.box_instance_matrix(np.eye(4), (2.0, 1.0, 0.5))
    corner = b @ np.array([0.5, 0.5, 0.5, 1.0])
    assert np.allclose(np.abs(corner), [1.0, 0.5, 2.0])  # UE extent (2, 1, 0.5) -> renderer (y, z, x)


def test_proxy_world_geometry():
    m = fake_carla.Map()
    inst = road_instances(m.generate_waypoints(2.0), 2.0)
    names = {i["name"] for i in inst}
    assert {"road_proxy_asphalt", "markings_white", "markings_yellow"} <= names
    road = next(i for i in inst if i["name"] == "road_proxy_asphalt")
    p = np.array(road["positions"]).reshape(-1, 3)
    assert np.allclose(p[:, 1], 0.0) and p[:, 0].min() == pytest.approx(-3.5) and p[:, 2].min() == pytest.approx(-200)
    env = environment_instances(fake_carla.World().get_environment_objects())
    assert len(env) == 1 and env[0]["seg_id"] == LABEL_IDS["Buildings"]  # roads skipped (built from waypoints)


def test_live_viewer_end_to_end(tmp_path):
    client = fake_carla.Client()
    viewer = spectral_live.run(fake_carla, client, cfg(tmp_path), max_frames=3, headless=True)
    s = viewer.s
    img = s.image()
    seg = img["seg_id"]
    h, w = seg.shape
    assert seg[h // 2 + 2, w // 2] == LABEL_IDS["Car"]           # lead vehicle ahead
    assert seg[h - 2, w // 2] == LABEL_IDS["Roads"]              # road below
    assert seg[1, w // 2] == 0xFFFFFFFF                           # sky
    assert (seg == LABEL_IDS["Buildings"]).sum() > 0             # building on the right
    assert (seg == LABEL_IDS["TrafficLight"]).sum() > 0
    assert s.live_objects() >= 5
    assert viewer.sync.stats["spawned"] == 1                      # lead only (proxy ego carries the camera)
    # Red lead car, blue-ish sky: spectral content is sensible.
    car = img["radiance"][seg == LABEL_IDS["Car"]].mean(axis=0)
    wl = list(s.wavelengths)
    assert car[wl.index(640)] > car[wl.index(460)]


def test_quality_mode_records_and_actor_removal(tmp_path):
    client = fake_carla.Client()
    c = cfg(tmp_path)
    rig = spectral_live.CarlaRig(fake_carla, client, c)
    session = sr.Session(spectral_live.base_scene(c), backend="cpu", base_dir=ROOT)
    sync = spectral_live.CarlaSync(session, rig.world, c, rig.camera, (96, 54))
    sync.build_static()
    v = spectral_live.Viewer(session, rig, sync, c, headless=True)
    assert v.step()
    v.quality, v.record = True, True
    assert v.step()
    assert session.spp == 8
    assert os.path.exists(os.path.join(c["save_dir"], f"frame_{v.frame:06d}.npz"))
    lead = next(a for a in rig.world.actors if a.type_id == "vehicle.audi.a2")
    lead.destroy()
    v.quality = False
    assert v.step()
    assert sync.stats["removed"] == 1
    # Traffic light turns green: its emissive proxy is swapped.
    tl = next(a for a in rig.world.actors if a.type_id.startswith("traffic."))
    tl.state = fake_carla.Label("Green")
    assert v.step()
    assert sync.traffic_lights[tl.id]["state"] == "Green"
    for mode in range(5):
        v.view = mode
        v.draw()
    rig.close()
