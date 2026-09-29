// Surface interaction reconstruction from a hit record.
#pragma once
#include "spectral/kernel/scene_view.h"

namespace spectral {

struct SurfaceInteraction {
  Vec3 p;      // world position
  Vec3 ng;     // geometric normal (from winding, NOT flipped towards the ray)
  Vec3 ns;     // interpolated shading normal, same hemisphere as ng
  Vec3 dpdu, dpdv;
  Vec2 uv;
  Vec3 wo;     // direction towards the previous vertex
  float t = 0.f;
  bool front = true;  // ray arrived from the side ng points to
  int32_t material = -1;
  uint32_t instance = 0, prim = 0;
};

SPECTRAL_FN void triangle_indices(const SceneView& s, const MeshRecord& m, uint32_t prim, uint32_t idx[3]) {
  const uint32_t* tri = s.indices + 3 * size_t(m.tri_offset + prim);
  idx[0] = m.vtx_offset + tri[0];
  idx[1] = m.vtx_offset + tri[1];
  idx[2] = m.vtx_offset + tri[2];
}

SPECTRAL_FN SurfaceInteraction make_interaction(const SceneView& s, uint32_t instance, uint32_t prim, float b1,
                                                float b2, Vec3 ray_dir, float t) {
  SurfaceInteraction si;
  const InstanceRecord& inst = s.instances[instance];
  const MeshRecord& mesh = s.meshes[inst.mesh];
  uint32_t idx[3];
  triangle_indices(s, mesh, prim, idx);
  float b0 = 1.f - b1 - b2;
  Vec3 p0 = inst.to_world.point(s.positions[idx[0]]);
  Vec3 p1 = inst.to_world.point(s.positions[idx[1]]);
  Vec3 p2 = inst.to_world.point(s.positions[idx[2]]);
  si.p = p0 * b0 + p1 * b1 + p2 * b2;
  Vec3 e1 = p1 - p0, e2 = p2 - p0;
  si.ng = normalize(cross(e1, e2));
  if (mesh.flags & kMeshHasNormals) {
    Vec3 n = s.normals[idx[0]] * b0 + s.normals[idx[1]] * b1 + s.normals[idx[2]] * b2;
    si.ns = normalize(inst.to_local.normal_from_inverse(n));
    if (!(length_sq(si.ns) > 0.f) || dot(si.ns, si.ns) != dot(si.ns, si.ns)) si.ns = si.ng;
    if (dot(si.ns, si.ng) < 0.f) si.ng = -si.ng;  // trust authored normals for orientation
  } else {
    si.ns = si.ng;
  }
  Vec2 uv0(0, 0), uv1(1, 0), uv2(1, 1);
  if (mesh.flags & kMeshHasUVs) {
    uv0 = s.uvs[idx[0]];
    uv1 = s.uvs[idx[1]];
    uv2 = s.uvs[idx[2]];
  }
  si.uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
  Vec2 duv1 = uv1 - uv0, duv2 = uv2 - uv0;
  float det = duv1.x * duv2.y - duv1.y * duv2.x;
  if (fabsf(det) > 1e-12f) {
    float inv = 1.f / det;
    si.dpdu = (e1 * duv2.y - e2 * duv1.y) * inv;
    si.dpdv = (e2 * duv1.x - e1 * duv2.x) * inv;
  } else {
    Frame f = Frame::from_z(si.ng);
    si.dpdu = f.x;
    si.dpdv = f.y;
  }
  si.wo = -normalize(ray_dir);
  si.front = dot(si.ng, si.wo) >= 0.f;
  si.material = inst.material;
  si.instance = instance;
  si.prim = prim;
  si.t = t;
  return si;
}

}  // namespace spectral
