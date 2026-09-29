// Procedural meshes used by tests and simple scenes.
#pragma once
#include "spectral/scene/scene.h"

namespace spectral {

MeshData make_sphere(float radius, int segments = 64, int rings = 32);
MeshData make_quad(float size_x, float size_z);                        // XZ plane, normal +Y, centred
MeshData make_box(float size_x, float size_y, float size_z);           // centred, outward normals
MeshData make_disk(float radius, int segments = 64);                   // XZ plane, normal +Y

}  // namespace spectral
