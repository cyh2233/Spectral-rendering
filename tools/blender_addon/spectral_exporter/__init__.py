"""Blender add-on: export the current scene for the spectral renderer.

Writes <name>.glb (geometry, glTF PBR materials, textures) and <name>.json (camera, instances,
segmentation ids, spectral material overrides, lights, sky, render settings). Optionally runs
`spectral_render` and loads the sRGB preview into Blender's image editor.

Per-object custom property:   seg_id (int)
Per-material custom properties (all optional, see docs/blender_export.md):
  spectral_reflectance, spectral_bsdf (pbr|dielectric|thin_dielectric), spectral_glass,
  spectral_ior, spectral_transmittance, spectral_internal_transmittance, spectral_thickness_mm,
  spectral_emission, spectral_emission_scale, spectral_roughness, spectral_metallic, spectral_specular
  Values are library spectrum names (e.g. "approx_asphalt_dry"), numbers, or JSON objects.
Per-light custom properties: spectral_emission (library name) or spectral_temperature (K).
"""
bl_info = {
    "name": "Spectral Renderer Exporter",
    "author": "spectral-rendering",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "File > Export > Spectral Scene (.json); Properties > Render > Spectral",
    "description": "Exports scenes (glTF + JSON) for the dense-band spectral path tracer",
    "category": "Import-Export",
}

import json
import math
import os
import re
import subprocess

try:
    import bpy
    from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
    from bpy_extras.io_utils import ExportHelper
    from mathutils import Matrix, Vector
except ImportError:  # allows importing the pure helpers outside Blender
    bpy = None

# Blender (Z up) -> glTF / renderer (Y up): (x, y, z) -> (x, z, -y)
AXIS = None if bpy is None else Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))

MATERIAL_KEYS = {
    "spectral_reflectance": "reflectance", "spectral_bsdf": "bsdf", "spectral_glass": "glass",
    "spectral_ior": "ior", "spectral_transmittance": "transmittance",
    "spectral_internal_transmittance": "internal_transmittance", "spectral_thickness_mm": "thickness_mm",
    "spectral_emission": "emission", "spectral_emission_scale": "emission_scale",
    "spectral_roughness": "roughness", "spectral_metallic": "metallic", "spectral_specular": "specular",
}


def _value(v):
    """Custom property value -> JSON value (strings starting with '{' or '[' are parsed as JSON)."""
    if hasattr(v, "to_list"):
        return v.to_list()
    if hasattr(v, "to_dict"):
        return v.to_dict()
    if isinstance(v, str) and v.strip()[:1] in "{[":
        return json.loads(v)
    return v


def _exact(name):
    return "^" + re.escape(name) + "$"


def _rows(m):
    return [[float(m[r][c]) for c in range(4)] for r in range(3)]


def camera_json(scene, cam_obj):
    cam = cam_obj.data
    r = scene.render
    w = int(r.resolution_x * r.resolution_percentage / 100)
    h = int(r.resolution_y * r.resolution_percentage / 100)
    fit = cam.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if w >= h else "VERTICAL"
        size = cam.sensor_width
    else:
        size = cam.sensor_width if fit == "HORIZONTAL" else cam.sensor_height
    if cam.type != "PERSP":
        raise ValueError("only perspective cameras are supported (pinhole)")
    if abs(cam.shift_x) > 1e-6 or abs(cam.shift_y) > 1e-6:
        print("[spectral] warning: lens shift is ignored")
    fov = 2.0 * math.degrees(math.atan(0.5 * size / cam.lens))
    loc, rot, _ = cam_obj.matrix_world.decompose()
    m = AXIS @ (Matrix.Translation(loc) @ rot.to_matrix().to_4x4())
    return {"type": "pinhole", "matrix": _rows(m), "fov_deg": fov,
            "fov_axis": "horizontal" if fit == "HORIZONTAL" else "vertical", "resolution": [w, h]}


def material_overrides(materials):
    out = []
    for mat in materials:
        o = {}
        for k, jk in MATERIAL_KEYS.items():
            if k in mat.keys():
                o[jk] = _value(mat[k])
        if o:
            o["match"] = _exact(mat.name)
            out.append(o)
    return out


def light_json(obj):
    L = obj.data
    to_world = AXIS @ obj.matrix_world
    pos = to_world.to_translation()
    shape = ({"type": "library", "name": obj["spectral_emission"]} if "spectral_emission" in obj.keys()
             else {"type": "blackbody", "temperature": float(obj.get("spectral_temperature", 6500.0))})
    if L.type in ("POINT", "SPOT"):
        # Blender point/spot power P [W] -> radiant intensity P / (4 pi) [W/sr] over 380-780 nm.
        emission = dict(shape, integral=float(L.energy) / (4 * math.pi))
        j = {"type": "point", "name": obj.name, "position": list(pos), "emission": emission}
        if L.type == "SPOT":
            d = (to_world.to_3x3() @ Vector((0, 0, -1))).normalized()
            outer = math.degrees(L.spot_size) / 2
            j.update(type="spot", direction=list(d), cone_outer_deg=outer, cone_inner_deg=outer * (1 - L.spot_blend))
        return j
    if L.type == "SUN":
        return {"_sun_direction": list((to_world.to_3x3() @ Vector((0, 0, 1))).normalized()),
                "_sun_irradiance": dict(shape, integral=float(L.energy))}
    print(f"[spectral] warning: {L.type} light '{obj.name}' skipped (use an emissive mesh for area lights)")
    return None


def export_scene(scene, filepath, selected_only=False, write_glb=True):
    """Exports `scene` to <filepath>.json and <stem>.glb. Returns the scene dict."""
    base = os.path.dirname(os.path.abspath(filepath))
    stem = os.path.splitext(os.path.basename(filepath))[0]
    glb = os.path.join(base, stem + ".glb")
    if write_glb:
        bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_selection=selected_only, export_apply=True,
                                  export_yup=True, export_cameras=False, export_lights=False, export_extras=False)
    if scene.camera is None:
        raise ValueError("scene has no active camera")
    objs = [o for o in scene.objects if (o.select_get() or not selected_only) and o.visible_get()]
    seg = {_exact(o.name): int(o["seg_id"]) for o in objs if "seg_id" in o.keys()}
    mats = {s.material for o in objs if o.type == "MESH" for s in o.material_slots if s.material}
    p = scene.spectral if hasattr(scene, "spectral") else None
    lights, sun = [], None
    for o in objs:
        if o.type == "LIGHT":
            j = light_json(o)
            if j and "_sun_direction" in j:
                sun = j
            elif j:
                lights.append(j)
    sky_model = p.sky_model if p else "simple"
    if sky_model != "none":
        sky = {"type": "sky", "model": sky_model}
        if sun:
            sky["sun"] = {"direction": sun["_sun_direction"]}
        if sky_model == "simple":
            sky["sky_radiance"] = {"type": "cie", "name": "D65", "scale": p.sky_scale if p else 0.0005}
            if sun:
                sky["sun"]["irradiance"] = sun["_sun_irradiance"]
        elif sky_model == "prague":
            sky.update(dataset=bpy.path.abspath(p.prague_dataset), visibility_km=p.visibility_km,
                       ground_albedo=p.ground_albedo)
        elif sky_model == "plugin":
            sky.update(library=bpy.path.abspath(p.plugin_library), options=json.loads(p.plugin_options or "{}"))
        lights.append(sky)
    doc = {
        "version": 2,
        "assets": {"scene": {"path": stem + ".glb"}},
        "camera": camera_json(scene, scene.camera),
        "instances": [{"asset": "scene", "seg_id": 0, "seg_id_by_node": seg}],
        "materials": {"use_default_overrides": bool(p.use_default_overrides) if p else False,
                      "overrides": material_overrides(sorted(mats, key=lambda m: m.name))},
        "lights": lights,
        "render": {"spp": p.spp if p else 64, "max_depth": p.max_depth if p else 8, "seed": p.seed if p else 0,
                   "backend": p.backend if p else "auto",
                   "bands": {"min": p.band_min if p else 380, "max": p.band_max if p else 1000,
                             "step": p.band_step if p else 5}},
        "output": {"exr": stem + ".exr", "npz": stem + ".npz", "preview_png": stem + "_srgb.png",
                   "metadata": {"source": "blender", "blend_file": bpy.data.filepath, "frame": scene.frame_current}},
    }
    with open(os.path.join(base, stem + ".json"), "w") as f:
        json.dump(doc, f, indent=2)
    return doc


if bpy is not None:

    class SpectralSettings(bpy.types.PropertyGroup):
        spp: IntProperty(name="Samples", default=64, min=1)
        max_depth: IntProperty(name="Max bounces", default=8, min=1)
        seed: IntProperty(name="Seed", default=0, min=0)
        band_min: FloatProperty(name="Min (nm)", default=380.0)
        band_max: FloatProperty(name="Max (nm)", default=1000.0)
        band_step: FloatProperty(name="Step (nm)", default=5.0, min=0.5)
        backend: EnumProperty(name="Backend", items=[("auto", "Auto", ""), ("cpu", "CPU", ""), ("cuda", "CUDA", "")])
        sky_model: EnumProperty(name="Sky", items=[("simple", "Simple (D65 + sun)", ""), ("prague", "Prague", ""),
                                                   ("plugin", "Plugin", ""), ("none", "None", "")])
        sky_scale: FloatProperty(name="Sky radiance scale", default=0.0005, min=0.0,
                                  description="Scale of CIE D65 (100 at 560 nm): 0.0005 ~ 0.05 W/(m^2 sr nm), clear sky")
        prague_dataset: StringProperty(name="Prague dataset", subtype="FILE_PATH")
        visibility_km: FloatProperty(name="Visibility (km)", default=59.4, min=20.0, max=131.8)
        ground_albedo: FloatProperty(name="Ground albedo", default=0.33, min=0.0, max=1.0)
        plugin_library: StringProperty(name="Sky plugin", subtype="FILE_PATH")
        plugin_options: StringProperty(name="Plugin options (JSON)", default="{}")
        use_default_overrides: BoolProperty(name="Default material spectra", default=False)

    class SpectralPreferences(bpy.types.AddonPreferences):
        bl_idname = __name__
        renderer: StringProperty(name="spectral_render executable", subtype="FILE_PATH")

        def draw(self, context):
            self.layout.prop(self, "renderer")

    class SPECTRAL_OT_export(bpy.types.Operator, ExportHelper):
        """Export the scene for the spectral renderer"""
        bl_idname = "export_scene.spectral"
        bl_label = "Export Spectral Scene"
        filename_ext = ".json"
        selected_only: BoolProperty(name="Selected only", default=False)

        def execute(self, context):
            try:
                export_scene(context.scene, self.filepath, self.selected_only)
            except Exception as e:  # report to the UI
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            self.report({"INFO"}, f"Exported {self.filepath}")
            return {"FINISHED"}

    class SPECTRAL_OT_render(bpy.types.Operator):
        """Export to the output folder, run spectral_render and show the sRGB preview"""
        bl_idname = "render.spectral"
        bl_label = "Render Spectral"

        def execute(self, context):
            prefs = context.preferences.addons[__name__].preferences
            exe = bpy.path.abspath(prefs.renderer)
            if not exe or not os.path.exists(exe):
                self.report({"ERROR"}, "Set the spectral_render executable in the add-on preferences")
                return {"CANCELLED"}
            out = bpy.path.abspath(context.scene.render.filepath) or bpy.app.tempdir
            os.makedirs(out, exist_ok=True)
            path = os.path.join(out, "spectral_scene.json")
            export_scene(context.scene, path)
            r = subprocess.run([exe, path], capture_output=True, text=True)
            if r.returncode != 0:
                self.report({"ERROR"}, r.stderr[-500:])
                return {"CANCELLED"}
            img = bpy.data.images.load(os.path.join(out, "spectral_scene_srgb.png"), check_existing=True)
            img.reload()
            self.report({"INFO"}, f"Rendered {path}")
            return {"FINISHED"}

    class SPECTRAL_PT_panel(bpy.types.Panel):
        bl_label = "Spectral"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "render"

        def draw(self, context):
            p = context.scene.spectral
            col = self.layout.column()
            for k in ("spp", "max_depth", "seed", "backend", "band_min", "band_max", "band_step", "sky_model"):
                col.prop(p, k)
            if p.sky_model == "simple":
                col.prop(p, "sky_scale")
            elif p.sky_model == "prague":
                for k in ("prague_dataset", "visibility_km", "ground_albedo"):
                    col.prop(p, k)
            elif p.sky_model == "plugin":
                col.prop(p, "plugin_library")
                col.prop(p, "plugin_options")
            col.prop(p, "use_default_overrides")
            col.operator("render.spectral")

    def _menu(self, context):
        self.layout.operator(SPECTRAL_OT_export.bl_idname, text="Spectral Scene (.json)")

    _classes = (SpectralSettings, SpectralPreferences, SPECTRAL_OT_export, SPECTRAL_OT_render, SPECTRAL_PT_panel)

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        bpy.types.Scene.spectral = bpy.props.PointerProperty(type=SpectralSettings)
        bpy.types.TOPBAR_MT_file_export.append(_menu)

    def unregister():
        bpy.types.TOPBAR_MT_file_export.remove(_menu)
        del bpy.types.Scene.spectral
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)
