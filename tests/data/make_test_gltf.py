#!/usr/bin/env python3
"""Generates tests/data/gltf/test_asset.gltf: a tiny asset exercising the glTF loader.

Contents:
  mesh "Panel": a unit quad in the XY plane (normal +Z) with UVs, material "M_Paint_Red"
               (base colour texture: 2x2 PNG, red/green/blue/white).
  mesh "Glass": a quad with KHR_materials_transmission (-> thin dielectric), material "M_Glass".
  nodes: "root" (translation [0, 1, 0]) with children "panel_a" (identity) and
         "panel_b" (translation [2, 0, 0], rotation 90 deg about +Y), and "glass" (translation [0,0,1]).
"""
import base64, json, struct, zlib, os

def png_rgba(w, h, pixels):
    raw = b"".join(b"\x00" + bytes(sum(pixels[y * w:(y + 1) * w], [])) for y in range(h))
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

pos = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]
nrm = [(0, 0, 1)] * 4
uv = [(0, 1), (1, 1), (1, 0), (0, 0)]
idx = [0, 1, 2, 0, 2, 3]
buf = b"".join(struct.pack("<3f", *p) for p in pos) + b"".join(struct.pack("<3f", *n) for n in nrm) + \
      b"".join(struct.pack("<2f", *t) for t in uv) + struct.pack("<6H", *idx) + b"\x00\x00"
img = png_rgba(2, 2, [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]])

gltf = {
    "asset": {"version": "2.0", "generator": "make_test_gltf.py"},
    "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior"],
    "buffers": [{"byteLength": len(buf), "uri": "data:application/octet-stream;base64," + base64.b64encode(buf).decode()}],
    "bufferViews": [
        {"buffer": 0, "byteOffset": 0, "byteLength": 48},
        {"buffer": 0, "byteOffset": 48, "byteLength": 48},
        {"buffer": 0, "byteOffset": 96, "byteLength": 32},
        {"buffer": 0, "byteOffset": 128, "byteLength": 12},
    ],
    "accessors": [
        {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [-0.5, -0.5, 0], "max": [0.5, 0.5, 0]},
        {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC2"},
        {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
    ],
    "images": [{"uri": "data:image/png;base64," + base64.b64encode(img).decode()}],
    "samplers": [{"magFilter": 9728, "minFilter": 9728, "wrapS": 33071, "wrapT": 33071}],
    "textures": [{"source": 0, "sampler": 0}],
    "materials": [
        {"name": "M_Paint_Red", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "baseColorTexture": {"index": 0},
                                                           "metallicFactor": 0.0, "roughnessFactor": 0.7}},
        {"name": "M_Glass", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "roughnessFactor": 0.0},
         "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.52}},
         "doubleSided": True},
    ],
    "meshes": [
        {"name": "Panel", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]},
        {"name": "Glass", "primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 3, "material": 1}]},
    ],
    "nodes": [
        {"name": "root", "translation": [0, 1, 0], "children": [1, 2, 3]},
        {"name": "panel_a", "mesh": 0},
        {"name": "panel_b", "mesh": 0, "translation": [2, 0, 0], "rotation": [0, 0.7071068, 0, 0.7071068]},
        {"name": "glass", "mesh": 1, "translation": [0, 0, 1]},
    ],
    "scenes": [{"nodes": [0]}],
    "scene": 0,
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gltf", "test_asset.gltf")
with open(out, "w") as f:
    json.dump(gltf, f, indent=1)
print("wrote", out)
