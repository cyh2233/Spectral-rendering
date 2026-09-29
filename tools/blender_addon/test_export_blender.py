"""Headless test of the Blender exporter (requires the `bpy` module, e.g. `pip install bpy==4.2.0`).

Builds a small road scene, exports it, renders it with spectral_render and checks that the
segmentation ids of objects appear where Blender's camera sees them.
Usage: python tools/blender_addon/test_export_blender.py <spectral_render exe> <output dir>
"""
import json
import math
import os
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spectral_exporter as se  # noqa: E402

exe, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
se.register()
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 96, 64, 100

def material(name, rgb, **props):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*rgb, 1)
    for k, v in props.items():
        m[k] = v
    return m

bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
road = bpy.context.object
road.name = "Road"
road["seg_id"] = 7
road.data.materials.append(material("M_Road", (0.1, 0.1, 0.1), spectral_reflectance="approx_asphalt_dry"))

bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 20, 1))  # 20 m ahead of the camera (+Y in Blender)
car = bpy.context.object
car.name = "Car"
car["seg_id"] = 10
car.data.materials.append(material("M_CarPaint", (0.8, 0.05, 0.05), spectral_reflectance="approx_car_paint_red"))

bpy.ops.mesh.primitive_cube_add(size=1, location=(6, 20, 0.5))  # to the right of the car
box = bpy.context.object
box.name = "Box"
box["seg_id"] = 11
box.data.materials.append(material("M_Box", (0.2, 0.6, 0.2)))

# Windshield: thin glass pane in front of the camera, raked 60 deg.
bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, 0.6, 1.2), rotation=(math.radians(30), 0, 0))
ws = bpy.context.object
ws.name = "Windshield"
ws.data.materials.append(material("M_Windshield", (1, 1, 1), spectral_bsdf="thin_dielectric",
                                  spectral_glass="soda_lime", spectral_transmittance="approx_windshield_green_5mm"))

bpy.ops.object.camera_add(location=(0, 0, 1.2), rotation=(math.radians(90), 0, 0))  # looks along +Y
scene.camera = bpy.context.object
scene.camera.data.lens = 35.0
bpy.ops.object.light_add(type="SUN", location=(0, 0, 10), rotation=(math.radians(40), 0, math.radians(30)))
bpy.context.object.data.energy = 400.0  # W/m^2 over 380-780 nm (physical daylight)
bpy.ops.object.light_add(type="POINT", location=(3, 10, 3))
bpy.context.object.data.energy = 500.0

scene.spectral.spp = 8
scene.spectral.band_step = 20.0
doc = se.export_scene(scene, os.path.join(out, "blender_test.json"))
subprocess.run([exe, os.path.join(out, "blender_test.json"), "--quiet"], check=True)
z = np.load(os.path.join(out, "blender_test.npz"))
seg = z["seg_id"]
H, W = seg.shape

def project(p):
    """Pixel of world point p (Blender coordinates) through Blender's camera."""
    from bpy_extras.object_utils import world_to_camera_view
    co = world_to_camera_view(scene, scene.camera, Vector(p))
    return int((1 - co.y) * H), int(co.x * W)

checks = {"Car": ((0, 20, 1), 10), "Box": ((6, 20, 0.5), 11), "Road": ((0, 8, 0), 7)}
ok = True
for name, (p, sid) in checks.items():
    r, c = project(p)
    got = int(seg[r, c])
    print(f"{name}: pixel ({r},{c}) seg {got} expected {sid}")
    ok &= got == sid
sky_r, sky_c = project((0, 1000, 200))
print("sky seg", hex(int(seg[sky_r, sky_c])))
ok &= int(seg[sky_r, sky_c]) == 0xFFFFFFFF
rad = z["radiance"]
wl = z["wavelengths"]
def band(lam):
    return int(np.argmin(np.abs(wl - lam)))
def mean_spec(sid):
    return rad[seg == sid].mean(axis=0)
car, green = mean_spec(10), mean_spec(11)
r_car = car[band(640)] / car[band(540)]
r_green = green[band(640)] / green[band(540)]
print(f"640/540 nm ratio: car {r_car:.2f}, green box {r_green:.2f}")
ok &= r_car > 1.5 and r_green < 0.8
print("OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
