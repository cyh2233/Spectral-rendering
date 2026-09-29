// Small vector/matrix library usable on host and device.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#include "spectral/macros.h"

namespace spectral {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kInvPi = 0.31830988618379067154f;
constexpr float kInf = 3.402823466e+38f;  // FLT_MAX used as "infinity" (device safe)

SPECTRAL_FN float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
SPECTRAL_FN float minf(float a, float b) { return a < b ? a : b; }
SPECTRAL_FN float maxf(float a, float b) { return a > b ? a : b; }
SPECTRAL_FN float sqr(float x) { return x * x; }
SPECTRAL_FN float lerpf(float t, float a, float b) { return a + t * (b - a); }
SPECTRAL_FN float safe_sqrt(float x) { return sqrtf(x > 0.f ? x : 0.f); }

struct Vec2 {
  float x = 0, y = 0;
  SPECTRAL_FN Vec2() {}
  SPECTRAL_FN Vec2(float a, float b) : x(a), y(b) {}
  SPECTRAL_FN Vec2 operator+(Vec2 o) const { return {x + o.x, y + o.y}; }
  SPECTRAL_FN Vec2 operator-(Vec2 o) const { return {x - o.x, y - o.y}; }
  SPECTRAL_FN Vec2 operator*(float s) const { return {x * s, y * s}; }
};

struct Vec3 {
  float x = 0, y = 0, z = 0;
  SPECTRAL_FN Vec3() {}
  SPECTRAL_FN Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
  SPECTRAL_FN explicit Vec3(float s) : x(s), y(s), z(s) {}
  SPECTRAL_FN float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
  SPECTRAL_FN float& operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
  SPECTRAL_FN Vec3 operator+(Vec3 o) const { return {x + o.x, y + o.y, z + o.z}; }
  SPECTRAL_FN Vec3 operator-(Vec3 o) const { return {x - o.x, y - o.y, z - o.z}; }
  SPECTRAL_FN Vec3 operator-() const { return {-x, -y, -z}; }
  SPECTRAL_FN Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
  SPECTRAL_FN Vec3 operator*(Vec3 o) const { return {x * o.x, y * o.y, z * o.z}; }
  SPECTRAL_FN Vec3 operator/(float s) const { float r = 1.f / s; return {x * r, y * r, z * r}; }
  SPECTRAL_FN Vec3& operator+=(Vec3 o) { x += o.x; y += o.y; z += o.z; return *this; }
  SPECTRAL_FN Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }
};
SPECTRAL_FN Vec3 operator*(float s, Vec3 v) { return v * s; }

struct Vec4 {
  float x = 0, y = 0, z = 0, w = 0;
  SPECTRAL_FN Vec4() {}
  SPECTRAL_FN Vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
};

SPECTRAL_FN float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
SPECTRAL_FN float absdot(Vec3 a, Vec3 b) { return fabsf(dot(a, b)); }
SPECTRAL_FN Vec3 cross(Vec3 a, Vec3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
SPECTRAL_FN float length(Vec3 a) { return sqrtf(dot(a, a)); }
SPECTRAL_FN float length_sq(Vec3 a) { return dot(a, a); }
SPECTRAL_FN Vec3 normalize(Vec3 a) {
  float l = length(a);
  return l > 0.f ? a / l : Vec3(0.f, 0.f, 1.f);
}
SPECTRAL_FN Vec3 vmin(Vec3 a, Vec3 b) { return {minf(a.x, b.x), minf(a.y, b.y), minf(a.z, b.z)}; }
SPECTRAL_FN Vec3 vmax(Vec3 a, Vec3 b) { return {maxf(a.x, b.x), maxf(a.y, b.y), maxf(a.z, b.z)}; }
SPECTRAL_FN float max_component(Vec3 a) { return maxf(a.x, maxf(a.y, a.z)); }
SPECTRAL_FN Vec3 face_forward(Vec3 n, Vec3 v) { return dot(n, v) < 0.f ? -n : n; }
SPECTRAL_FN Vec3 reflect(Vec3 wo, Vec3 n) { return -wo + n * (2.f * dot(wo, n)); }

// Orthonormal basis (Duff et al. 2017). z = n.
struct Frame {
  Vec3 x, y, z;
  SPECTRAL_FN Frame() : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
  SPECTRAL_FN static Frame from_z(Vec3 n) {
    Frame f;
    float sign = copysignf(1.f, n.z);
    float a = -1.f / (sign + n.z);
    float b = n.x * n.y * a;
    f.x = Vec3(1.f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    f.y = Vec3(b, sign + n.y * n.y * a, -n.y);
    f.z = n;
    return f;
  }
  // Build from normal and an approximate tangent (Gram-Schmidt).
  SPECTRAL_FN static Frame from_zx(Vec3 n, Vec3 t) {
    Vec3 tx = t - n * dot(n, t);
    float l = length(tx);
    if (!(l > 1e-8f)) return from_z(n);
    Frame f;
    f.z = n;
    f.x = tx / l;
    f.y = cross(f.z, f.x);
    return f;
  }
  SPECTRAL_FN Vec3 to_local(Vec3 v) const { return {dot(v, x), dot(v, y), dot(v, z)}; }
  SPECTRAL_FN Vec3 to_world(Vec3 v) const { return x * v.x + y * v.y + z * v.z; }
};

// Affine transform stored as 3x4 row-major matrix (last row implicit 0 0 0 1).
struct Affine {
  float m[3][4];
  SPECTRAL_FN static Affine identity() {
    Affine a;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) a.m[r][c] = (r == c) ? 1.f : 0.f;
    return a;
  }
  SPECTRAL_FN Vec3 point(Vec3 p) const {
    return {m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3],
            m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3],
            m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3]};
  }
  SPECTRAL_FN Vec3 vector(Vec3 v) const {
    return {m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z};
  }
  // Transform a normal with the inverse-transpose; `this` must be the INVERSE transform.
  SPECTRAL_FN Vec3 normal_from_inverse(Vec3 n) const {
    return {m[0][0] * n.x + m[1][0] * n.y + m[2][0] * n.z,
            m[0][1] * n.x + m[1][1] * n.y + m[2][1] * n.z,
            m[0][2] * n.x + m[1][2] * n.y + m[2][2] * n.z};
  }
};

SPECTRAL_FN Affine compose(const Affine& a, const Affine& b) {  // a * b
  Affine r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      float s = (j == 3) ? a.m[i][3] : 0.f;
      for (int k = 0; k < 3; ++k) s += a.m[i][k] * b.m[k][j];
      r.m[i][j] = s;
    }
  }
  return r;
}

SPECTRAL_FN Affine inverse(const Affine& a) {
  // Invert 3x3 part via cofactors (double precision for robustness), then translation.
  double m00 = a.m[0][0], m01 = a.m[0][1], m02 = a.m[0][2];
  double m10 = a.m[1][0], m11 = a.m[1][1], m12 = a.m[1][2];
  double m20 = a.m[2][0], m21 = a.m[2][1], m22 = a.m[2][2];
  double c00 = m11 * m22 - m12 * m21, c01 = m02 * m21 - m01 * m22, c02 = m01 * m12 - m02 * m11;
  double c10 = m12 * m20 - m10 * m22, c11 = m00 * m22 - m02 * m20, c12 = m02 * m10 - m00 * m12;
  double c20 = m10 * m21 - m11 * m20, c21 = m01 * m20 - m00 * m21, c22 = m00 * m11 - m01 * m10;
  double det = m00 * c00 + m01 * c10 + m02 * c20;
  double id = det != 0.0 ? 1.0 / det : 0.0;
  Affine r;
  double inv[3][3] = {{c00 * id, c01 * id, c02 * id}, {c10 * id, c11 * id, c12 * id}, {c20 * id, c21 * id, c22 * id}};
  for (int i = 0; i < 3; ++i) {
    double t = 0;
    for (int k = 0; k < 3; ++k) {
      r.m[i][k] = float(inv[i][k]);
      t += inv[i][k] * a.m[k][3];
    }
    r.m[i][3] = float(-t);
  }
  return r;
}

struct AABB {
  Vec3 lo = Vec3(kInf), hi = Vec3(-kInf);
  SPECTRAL_FN void expand(Vec3 p) { lo = vmin(lo, p); hi = vmax(hi, p); }
  SPECTRAL_FN void expand(const AABB& b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
  SPECTRAL_FN Vec3 center() const { return (lo + hi) * 0.5f; }
  SPECTRAL_FN Vec3 extent() const { return hi - lo; }
  SPECTRAL_FN bool valid() const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
  SPECTRAL_FN float area() const {
    if (!valid()) return 0.f;
    Vec3 e = extent();
    return 2.f * (e.x * e.y + e.y * e.z + e.z * e.x);
  }
};

SPECTRAL_FN AABB transform_aabb(const Affine& t, const AABB& b) {
  AABB r;
  for (int i = 0; i < 8; ++i) {
    Vec3 p((i & 1) ? b.hi.x : b.lo.x, (i & 2) ? b.hi.y : b.lo.y, (i & 4) ? b.hi.z : b.lo.z);
    r.expand(t.point(p));
  }
  return r;
}

SPECTRAL_FN int32_t float_as_int(float f) {
#if SPECTRAL_DEVICE_CODE
  return __float_as_int(f);
#else
  int32_t i;
  memcpy(&i, &f, 4);
  return i;
#endif
}
SPECTRAL_FN float int_as_float(int32_t i) {
#if SPECTRAL_DEVICE_CODE
  return __int_as_float(i);
#else
  float f;
  memcpy(&f, &i, 4);
  return f;
#endif
}

// Robust ray origin offset (Waechter & Binder, Ray Tracing Gems ch. 6).
SPECTRAL_FN Vec3 offset_ray_origin(Vec3 p, Vec3 n) {
  const float origin = 1.f / 32.f, float_scale = 1.f / 65536.f, int_scale = 256.f;
  int32_t oi[3] = {int32_t(int_scale * n.x), int32_t(int_scale * n.y), int32_t(int_scale * n.z)};
  float pi[3];
  for (int k = 0; k < 3; ++k) {
    float pk = p[k];
    pi[k] = int_as_float(float_as_int(pk) + ((pk < 0.f) ? -oi[k] : oi[k]));
  }
  return {fabsf(p.x) < origin ? p.x + float_scale * n.x : pi[0],
          fabsf(p.y) < origin ? p.y + float_scale * n.y : pi[1],
          fabsf(p.z) < origin ? p.z + float_scale * n.z : pi[2]};
}

// Spherical direction helpers. World convention: +Y up.
SPECTRAL_FN Vec3 spherical_direction(float sin_theta, float cos_theta, float phi) {
  return {sin_theta * cosf(phi), cos_theta, sin_theta * sinf(phi)};
}

}  // namespace spectral
