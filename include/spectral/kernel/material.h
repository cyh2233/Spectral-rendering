// Material evaluation: textures + spectral overrides -> shading parameters.
#pragma once
#include "spectral/kernel/interaction.h"
#include "spectral/kernel/texture.h"
#include "spectral/kernel/uplift.h"

namespace spectral {

struct ShadingParams {
  int32_t type = kMatPbr;
  Spectrum albedo;  // PBR: diffuse albedo / metallic F0
  float metallic = 0.f, roughness = 0.5f;
  float specular = 1.f;
  float ior = 1.5f;
  const float* ior_bands = nullptr;    // per-band IOR (Fresnel) or null
  const float* trans_bands = nullptr;  // thin dielectric internal transmittance or null
  Frame frame;                         // shading frame (z = shading normal, same side as ng)
};

SPECTRAL_FN float material_alpha(const SceneView& s, const MaterialRecord& m, Vec2 uv) {
  float a = m.base_color[3];
  if (m.base_color_tex >= 0) a *= sample_texture(s, m.base_color_tex, uv).w;
  return a;
}

// Alpha-mask test used by traversal; returns true if the hit is opaque.
SPECTRAL_FN bool alpha_test(const SceneView& s, uint32_t instance, uint32_t prim, float b1, float b2) {
  const InstanceRecord& inst = s.instances[instance];
  if (inst.material < 0) return true;
  const MaterialRecord& m = s.materials[inst.material];
  if (m.alpha_mode != kAlphaMask) return true;
  const MeshRecord& mesh = s.meshes[inst.mesh];
  if (!(mesh.flags & kMeshHasUVs)) return m.base_color[3] >= m.alpha_cutoff;
  uint32_t idx[3];
  triangle_indices(s, mesh, prim, idx);
  Vec2 uv = s.uvs[idx[0]] * (1.f - b1 - b2) + s.uvs[idx[1]] * b1 + s.uvs[idx[2]] * b2;
  return material_alpha(s, m, uv) >= m.alpha_cutoff;
}

SPECTRAL_FN bool is_glass(const MaterialRecord& m) { return m.type == kMatDielectric || m.type == kMatThinDielectric; }

SPECTRAL_FN void evaluate_material(const SceneView& s, const SurfaceInteraction& si, ShadingParams& sp) {
  const MaterialRecord& m = s.materials[si.material];
  sp.type = m.type;
  sp.ior = m.ior;
  sp.ior_bands = m.ior_spec >= 0 ? s.spectra + m.ior_spec : nullptr;
  sp.trans_bands = m.transmittance_spec >= 0 ? s.spectra + m.transmittance_spec : nullptr;
  sp.metallic = m.metallic;
  sp.roughness = m.roughness;
  sp.specular = m.specular;

  // Shading frame with optional normal map.
  Vec3 n = si.ns;
  if (m.normal_tex >= 0) {
    Vec4 t = sample_texture(s, m.normal_tex, si.uv);
    float nx = (2.f * t.x - 1.f) * m.normal_scale, ny = (2.f * t.y - 1.f) * m.normal_scale, nz = 2.f * t.z - 1.f;
    Frame tf = Frame::from_zx(n, si.dpdu);
    float handed = dot(cross(n, si.dpdu), si.dpdv) < 0.f ? -1.f : 1.f;
    Vec3 mapped = normalize(tf.x * nx + tf.y * (ny * handed) + tf.z * nz);
    if (dot(mapped, si.ng) * dot(n, si.ng) > 0.f) n = mapped;
  }
  sp.frame = Frame::from_zx(n, si.dpdu);

  if (m.type != kMatPbr) {
    sp.albedo.n = s.grid.n;
    sp.albedo.fill(1.f);
    return;
  }
  if (m.metal_rough_tex >= 0) {
    Vec4 t = sample_texture(s, m.metal_rough_tex, si.uv);
    sp.roughness *= t.y;
    sp.metallic *= t.z;
  }
  if (m.reflectance_spec >= 0) {
    sp.albedo.load(s.spectra + m.reflectance_spec, s.grid.n);
  } else {
    float r = m.base_color[0], g = m.base_color[1], b = m.base_color[2];
    if (m.base_color_tex >= 0) {
      Vec4 t = sample_texture(s, m.base_color_tex, si.uv);
      r *= t.x;
      g *= t.y;
      b *= t.z;
    }
    uplift_reflectance(s, r, g, b, sp.albedo);
  }
}

SPECTRAL_FN bool material_emits(const MaterialRecord& m) {
  return m.emission_spec >= 0 || m.emissive[0] > 0.f || m.emissive[1] > 0.f || m.emissive[2] > 0.f;
}

// Emitted radiance at a surface point (independent of direction, one- or two-sided).
SPECTRAL_FN void evaluate_emission(const SceneView& s, const MaterialRecord& m, Vec2 uv, Spectrum& Le) {
  Le.n = s.grid.n;
  if (m.emission_spec >= 0) {
    Le.load(s.spectra + m.emission_spec, s.grid.n);
    float sc = m.emission_scale;
    if (m.emissive_tex >= 0) {
      Vec4 t = sample_texture(s, m.emissive_tex, uv);
      sc *= 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
    }
    Le *= sc;
    return;
  }
  float r = m.emissive[0], g = m.emissive[1], b = m.emissive[2];
  if (m.emissive_tex >= 0) {
    Vec4 t = sample_texture(s, m.emissive_tex, uv);
    r *= t.x;
    g *= t.y;
    b *= t.z;
  }
  float k = m.emissive_strength * m.emission_scale;
  uplift_illuminant(s, r * k, g * k, b * k, Le);
}

}  // namespace spectral
