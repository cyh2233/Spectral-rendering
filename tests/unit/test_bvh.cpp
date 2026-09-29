#include "doctest.h"
#include "backend/cpu/bvh.h"
#include "spectral/kernel/rng.h"
#include "spectral/kernel/sampling.h"
#include "spectral/scene/primitives.h"
#include "spectral/scene/scene.h"

using namespace spectral;

namespace {
// Brute force closest hit over all instances/triangles (Moller-Trumbore in double precision).
bool brute_force(const SceneView& s, const Ray& r, Hit* best) {
  bool found = false;
  for (uint32_t ii = 0; ii < s.n_instances; ++ii) {
    const InstanceRecord& inst = s.instances[ii];
    const MeshRecord& m = s.meshes[inst.mesh];
    for (uint32_t t = 0; t < m.tri_count; ++t) {
      uint32_t idx[3];
      triangle_indices(s, m, t, idx);
      Vec3 p0 = inst.to_world.point(s.positions[idx[0]]), p1 = inst.to_world.point(s.positions[idx[1]]),
           p2 = inst.to_world.point(s.positions[idx[2]]);
      double e1[3] = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z}, e2[3] = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
      double d[3] = {r.d.x, r.d.y, r.d.z};
      double pv[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
      double det = e1[0] * pv[0] + e1[1] * pv[1] + e1[2] * pv[2];
      if (std::fabs(det) < 1e-14) continue;
      double tv[3] = {r.o.x - p0.x, r.o.y - p0.y, r.o.z - p0.z};
      double u = (tv[0] * pv[0] + tv[1] * pv[1] + tv[2] * pv[2]) / det;
      if (u < 0 || u > 1) continue;
      double q[3] = {tv[1] * e1[2] - tv[2] * e1[1], tv[2] * e1[0] - tv[0] * e1[2], tv[0] * e1[1] - tv[1] * e1[0]};
      double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / det;
      if (v < 0 || u + v > 1) continue;
      double tt = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) / det;
      if (tt <= 0 || tt >= best->t) continue;
      best->t = float(tt);
      best->instance = ii;
      best->prim = t;
      found = true;
    }
  }
  return found;
}
}  // namespace

TEST_CASE("two-level BVH agrees with brute force") {
  Scene scene;
  Rng rng = Rng::for_pixel_sample(3, 1, 4, 1);
  // Random triangle soup mesh + spheres with random transforms.
  MeshData soup;
  for (int t = 0; t < 400; ++t) {
    Vec3 c(rng.next1d() * 4 - 2, rng.next1d() * 4 - 2, rng.next1d() * 4 - 2);
    for (int k = 0; k < 3; ++k) {
      soup.positions.push_back(c + Vec3(rng.next1d() - 0.5f, rng.next1d() - 0.5f, rng.next1d() - 0.5f) * 0.6f);
      soup.indices.push_back(uint32_t(soup.positions.size() - 1));
    }
  }
  MaterialRecord mat;
  int32_t mi = scene.add_material(mat, "m");
  soup.material = mi;
  uint32_t soup_mesh = scene.add_mesh(soup);
  MeshData sph = make_sphere(0.7f, 24, 12);
  sph.material = mi;
  uint32_t sph_mesh = scene.add_mesh(sph);
  scene.add_instance(soup_mesh, Affine::identity());
  for (int i = 0; i < 6; ++i) {
    Affine xf = Affine::identity();
    float s = 0.5f + rng.next1d();
    xf.m[0][0] = s;
    xf.m[1][1] = s * 1.3f;
    xf.m[2][2] = s * 0.7f;
    xf.m[0][1] = 0.3f * rng.next1d();
    xf.m[0][3] = rng.next1d() * 6 - 3;
    xf.m[1][3] = rng.next1d() * 6 - 3;
    xf.m[2][3] = rng.next1d() * 6 - 3;
    scene.add_instance(i % 2 ? sph_mesh : soup_mesh, xf);
  }
  scene.finalize();
  SceneView v = scene.view();
  cpu::CpuAccel accel;
  accel.build(v);
  cpu::CpuIntersector isect{&v, &accel};
  int hits = 0, mismatches = 0;
  for (int i = 0; i < 20000; ++i) {
    Ray r;
    r.o = Vec3(rng.next1d() * 12 - 6, rng.next1d() * 12 - 6, rng.next1d() * 12 - 6);
    r.d = sample_uniform_sphere(rng.next2d());
    Hit a, b;
    bool ha = isect.closest(r, kInf, &a);
    bool hb = brute_force(v, r, &b);
    if (ha != hb) {
      ++mismatches;
      continue;
    }
    if (ha) {
      ++hits;
      if (std::fabs(a.t - b.t) > 1e-4f * std::max(1.f, b.t)) ++mismatches;
    }
    CHECK(isect.occluded(r, kInf) == hb);
  }
  CHECK(hits > 1000);
  CHECK(mismatches <= 2);  // edge-grazing rays may legitimately differ
}

TEST_CASE("primitive meshes have consistent winding and normals") {
  for (const MeshData& m : {make_sphere(1.f, 16, 8), make_box(1.f, 2.f, 3.f), make_quad(2.f, 1.f), make_disk(1.f, 12)}) {
    for (size_t t = 0; t < m.indices.size(); t += 3) {
      Vec3 p0 = m.positions[m.indices[t]], p1 = m.positions[m.indices[t + 1]], p2 = m.positions[m.indices[t + 2]];
      Vec3 ng = cross(p1 - p0, p2 - p0);
      if (length(ng) < 1e-8f) continue;
      Vec3 n = m.normals[m.indices[t]] + m.normals[m.indices[t + 1]] + m.normals[m.indices[t + 2]];
      CHECK(dot(ng, n) > 0.f);
    }
  }
}
