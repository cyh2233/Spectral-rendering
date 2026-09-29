#include "spectral/scene/primitives.h"

#include <cmath>

namespace spectral {

MeshData make_sphere(float r, int seg, int rings) {
  MeshData m;
  m.name = "sphere";
  for (int j = 0; j <= rings; ++j) {
    float v = float(j) / rings, theta = v * kPi;
    for (int i = 0; i <= seg; ++i) {
      float u = float(i) / seg, phi = u * 2.f * kPi;
      Vec3 n(std::sin(theta) * std::cos(phi), std::cos(theta), -std::sin(theta) * std::sin(phi));
      m.positions.push_back(n * r);
      m.normals.push_back(n);
      m.uvs.push_back(Vec2(u, v));
    }
  }
  for (int j = 0; j < rings; ++j)
    for (int i = 0; i < seg; ++i) {
      uint32_t a = j * (seg + 1) + i, b = a + seg + 1;
      // Counter-clockwise when seen from outside.
      if (j != 0) m.indices.insert(m.indices.end(), {a, b, a + 1});
      if (j != rings - 1) m.indices.insert(m.indices.end(), {a + 1, b, b + 1});
    }
  return m;
}

MeshData make_quad(float sx, float sz) {
  MeshData m;
  m.name = "quad";
  float hx = 0.5f * sx, hz = 0.5f * sz;
  m.positions = {Vec3(-hx, 0, -hz), Vec3(-hx, 0, hz), Vec3(hx, 0, hz), Vec3(hx, 0, -hz)};
  m.normals.assign(4, Vec3(0, 1, 0));
  m.uvs = {Vec2(0, 0), Vec2(0, 1), Vec2(1, 1), Vec2(1, 0)};
  m.indices = {0, 1, 2, 0, 2, 3};
  return m;
}

MeshData make_box(float sx, float sy, float sz) {
  MeshData m;
  m.name = "box";
  Vec3 h(0.5f * sx, 0.5f * sy, 0.5f * sz);
  const Vec3 normals[6] = {Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1)};
  for (int f = 0; f < 6; ++f) {
    Vec3 n = normals[f];
    Frame fr = Frame::from_z(n);
    uint32_t base = uint32_t(m.positions.size());
    const float cs[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (int k = 0; k < 4; ++k) {
      Vec3 p = n + fr.x * cs[k][0] + fr.y * cs[k][1];
      m.positions.push_back(Vec3(p.x * h.x, p.y * h.y, p.z * h.z));
      m.normals.push_back(n);
      m.uvs.push_back(Vec2(0.5f * (cs[k][0] + 1), 0.5f * (cs[k][1] + 1)));
    }
    // Frame (x, y, n) is right-handed, so (0,1,2) is CCW around n.
    m.indices.insert(m.indices.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
  }
  return m;
}

MeshData make_disk(float r, int seg) {
  MeshData m;
  m.name = "disk";
  m.positions.push_back(Vec3(0, 0, 0));
  m.normals.push_back(Vec3(0, 1, 0));
  m.uvs.push_back(Vec2(0.5f, 0.5f));
  for (int i = 0; i <= seg; ++i) {
    float a = 2.f * kPi * float(i) / seg;
    m.positions.push_back(Vec3(r * std::cos(a), 0, -r * std::sin(a)));
    m.normals.push_back(Vec3(0, 1, 0));
    m.uvs.push_back(Vec2(0.5f + 0.5f * std::cos(a), 0.5f + 0.5f * std::sin(a)));
  }
  for (int i = 1; i <= seg; ++i) m.indices.insert(m.indices.end(), {0u, uint32_t(i), uint32_t(i + 1)});
  return m;
}

}  // namespace spectral
