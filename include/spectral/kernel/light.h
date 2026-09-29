// Light sampling and evaluation. All returned pdfs are solid-angle pdfs that do NOT
// include the light-selection probability (see light_pmf()).
#pragma once
#include "spectral/kernel/material.h"
#include "spectral/kernel/sampling.h"

namespace spectral {

SPECTRAL_FN float light_pmf(const SceneView& s, int32_t light) {
  return (light >= 0 && s.light_alias) ? s.light_alias[light].pmf : 0.f;
}

// ---------------------------------------------------------------- environment table
SPECTRAL_FN Vec2 env_dir_to_uv(const EnvView& e, Vec3 d) {
  float theta = acosf(clampf(d.y, -1.f, 1.f));
  float phi = atan2f(d.z, d.x) - e.rotation;
  phi = phi - 2.f * kPi * floorf(phi / (2.f * kPi));
  return {phi / (2.f * kPi), theta / kPi};
}
SPECTRAL_FN Vec3 env_uv_to_dir(const EnvView& e, Vec2 uv) {
  float theta = uv.y * kPi, phi = uv.x * 2.f * kPi + e.rotation;
  return spherical_direction(sinf(theta), cosf(theta), phi);
}
SPECTRAL_FN void env_lookup(const SceneView& s, Vec3 d, Spectrum& L) {
  const EnvView& e = s.env;
  L.n = s.grid.n;
  if (!e.present) {
    L.fill(0.f);
    return;
  }
  Vec2 uv = env_dir_to_uv(e, d);
  int x = int(uv.x * e.width), y = int(uv.y * e.height);
  x = x < 0 ? 0 : (x >= e.width ? e.width - 1 : x);
  y = y < 0 ? 0 : (y >= e.height ? e.height - 1 : y);
  const float* px = e.data + (size_t(y) * e.width + x) * size_t(s.grid.n);
  for (int i = 0; i < L.n; ++i) L.v[i] = px[i] * e.scale;
}
SPECTRAL_FN float env_pdf(const EnvView& e, Vec3 d) {
  if (!e.present) return 0.f;
  Vec2 uv = env_dir_to_uv(e, d);
  float sin_t = sinf(uv.y * kPi);
  if (sin_t <= 0.f) return 0.f;
  return e.dist.pdf(uv) / (2.f * kPi * kPi * sin_t);
}

// ---------------------------------------------------------------- triangle lights
SPECTRAL_FN void triangle_world(const SceneView& s, uint32_t instance, uint32_t prim, Vec3 p[3]) {
  const InstanceRecord& inst = s.instances[instance];
  uint32_t idx[3];
  triangle_indices(s, s.meshes[inst.mesh], prim, idx);
  for (int k = 0; k < 3; ++k) p[k] = inst.to_world.point(s.positions[idx[k]]);
}

// pdf (solid angle, from `ref`) of hitting triangle light at point `p_light` with normal `n_light`.
SPECTRAL_FN float triangle_light_pdf(const LightRecord& l, Vec3 ref, Vec3 p_light, Vec3 n_light) {
  Vec3 d = p_light - ref;
  float dist2 = length_sq(d);
  float cos_l = absdot(n_light, d) / sqrtf(dist2);
  if (cos_l <= 1e-7f || l.area <= 0.f) return 0.f;
  return dist2 / (cos_l * l.area);
}

struct LightSample {
  Vec3 wi;               // direction from the reference point towards the light
  float dist = kInf;     // distance to the light point (kInf for lights at infinity)
  float pdf = 0.f;       // solid angle pdf (1 for delta lights)
  bool is_delta = false;
  Vec3 p_light, n_light; // for finite-area lights
};

// Samples a point on light `li` as seen from `ref`. Radiance arriving at ref is written to Li.
SPECTRAL_FN bool sample_light(const SceneView& s, int32_t li, Vec3 ref, Vec2 u, LightSample& ls, Spectrum& Li) {
  const LightRecord& l = s.lights[li];
  Li.n = s.grid.n;
  switch (l.type) {
    case kLightPoint:
    case kLightSpot: {
      Vec3 d = l.position - ref;
      float dist2 = length_sq(d);
      if (dist2 <= 0.f) return false;
      ls.dist = sqrtf(dist2);
      ls.wi = d / ls.dist;
      ls.pdf = 1.f;
      ls.is_delta = true;
      float k = l.scale / dist2;
      if (l.type == kLightSpot) {
        float c = dot(-ls.wi, l.direction);
        if (c <= l.cos_outer) return false;
        if (c < l.cos_inner) {
          float t = (c - l.cos_outer) / (l.cos_inner - l.cos_outer);
          k *= t * t * (3.f - 2.f * t);
        }
      }
      Li.load(s.spectra + l.spectrum, s.grid.n);
      Li *= k;
      return true;
    }
    case kLightDirectional: {
      ls.wi = -l.direction;
      ls.dist = kInf;
      ls.pdf = 1.f;
      ls.is_delta = true;
      Li.load(s.spectra + l.spectrum, s.grid.n);
      Li *= l.scale;
      return true;
    }
    case kLightSun: {
      Frame f = Frame::from_z(l.direction);
      ls.wi = f.to_world(sample_uniform_cone(u, l.cos_max));
      ls.dist = kInf;
      ls.pdf = uniform_cone_pdf(l.cos_max);
      ls.is_delta = false;
      Li.load(s.spectra + l.spectrum, s.grid.n);
      Li *= l.scale;
      return true;
    }
    case kLightTriangle: {
      Vec3 p[3];
      triangle_world(s, l.instance, l.prim, p);
      Vec2 b = sample_uniform_triangle(u);
      Vec3 pl = p[0] * (1.f - b.x - b.y) + p[1] * b.x + p[2] * b.y;
      Vec3 n = normalize(cross(p[1] - p[0], p[2] - p[0]));
      Vec3 d = pl - ref;
      float dist2 = length_sq(d);
      if (dist2 <= 0.f) return false;
      ls.dist = sqrtf(dist2);
      ls.wi = d / ls.dist;
      const InstanceRecord& inst = s.instances[l.instance];
      const MaterialRecord& m = s.materials[inst.material];
      float cos_signed = -dot(n, ls.wi);
      if (!m.double_sided && cos_signed <= 0.f) return false;
      float cos_l = fabsf(cos_signed);
      if (cos_l <= 1e-7f) return false;
      ls.pdf = dist2 / (cos_l * l.area);
      ls.is_delta = false;
      ls.p_light = pl;
      ls.n_light = n;
      SurfaceInteraction si = make_interaction(s, l.instance, l.prim, b.x, b.y, ls.wi, ls.dist);
      evaluate_emission(s, m, si.uv, Li);
      return true;
    }
    case kLightEnv: {
      float pdf_uv;
      Vec2 uv = s.env.dist.sample(u, &pdf_uv);
      float sin_t = sinf(uv.y * kPi);
      if (pdf_uv <= 0.f || sin_t <= 0.f) return false;
      ls.wi = env_uv_to_dir(s.env, uv);
      ls.pdf = pdf_uv / (2.f * kPi * kPi * sin_t);
      ls.dist = kInf;
      ls.is_delta = false;
      env_lookup(s, ls.wi, Li);
      return true;
    }
  }
  return false;
}

// Sun disk radiance along direction d (0 outside the disk).
SPECTRAL_FN bool sun_lookup(const SceneView& s, Vec3 d, Spectrum& L) {
  if (s.sun_light < 0) return false;
  const LightRecord& l = s.lights[s.sun_light];
  if (dot(d, l.direction) < l.cos_max) return false;
  L.load(s.spectra + l.spectrum, s.grid.n);
  L *= l.scale;
  return true;
}

}  // namespace spectral
