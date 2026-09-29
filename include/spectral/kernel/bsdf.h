// BSDFs in the local shading frame (z = shading normal).
//  - PBR: glTF metallic-roughness (Lambert diffuse + GGX specular, Schlick Fresnel).
//  - Dielectric: smooth interface; refraction direction from the reference IOR,
//    per-band Fresnel from ior_bands (no angular dispersion in the renderer).
//  - Thin dielectric: two-interface slab collapsed onto one surface (windows).
// All sampling pdfs are wavelength independent, so one path serves every band.
#pragma once
#include "spectral/kernel/material.h"
#include "spectral/kernel/sampling.h"

namespace spectral {

// Unpolarized Fresnel reflectance for a dielectric interface.
// cos_i > 0 on the incident side; eta = n_transmitted / n_incident.
SPECTRAL_FN float fresnel_dielectric(float cos_i, float eta) {
  cos_i = clampf(cos_i, 0.f, 1.f);
  float sin2_t = (1.f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.f) return 1.f;
  float cos_t = safe_sqrt(1.f - sin2_t);
  float rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
  float rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
  return 0.5f * (rs * rs + rp * rp);
}

SPECTRAL_FN float schlick(float f0, float cos_t) {
  float m = clampf(1.f - cos_t, 0.f, 1.f);
  float m2 = m * m;
  return f0 + (1.f - f0) * m2 * m2 * m;
}

SPECTRAL_FN bool refract_dir(Vec3 wi, Vec3 n, float eta, Vec3* wt) {
  // wi, n on the same side; eta = n_t / n_i.
  float cos_i = dot(n, wi);
  float sin2_t = maxf(0.f, 1.f - cos_i * cos_i) / (eta * eta);
  if (sin2_t >= 1.f) return false;
  float cos_t = safe_sqrt(1.f - sin2_t);
  *wt = -wi / eta + n * (cos_i / eta - cos_t);
  return true;
}

enum BsdfEvent : int32_t {
  kEventNone = 0,
  kEventDiffuseOrGlossy = 1,
  kEventSpecularReflect = 2,
  kEventSpecularTransmit = 3,  // direction changes (dielectric refraction)
  kEventPassThrough = 4        // direction unchanged (thin dielectric transmission)
};

struct BsdfSample {
  Vec3 wi;              // world space
  float pdf = 0.f;      // solid-angle pdf (1 for delta events)
  int32_t event = kEventNone;
};

// ---------------------------------------------------------------- PBR (metallic-roughness)
SPECTRAL_FN float pbr_alpha(const ShadingParams& sp) { return maxf(1e-3f, sp.roughness * sp.roughness); }
SPECTRAL_FN float pbr_spec_prob(const ShadingParams& sp) {
  float m = clampf(sp.metallic, 0.f, 1.f);
  float base = sp.specular > 0.f ? 0.25f : 0.f;  // no specular lobe to sample for pure Lambert
  return base + (1.f - base) * m;
}

// f(wo, wi) for local directions with wo.z > 0 (cosine NOT included).
SPECTRAL_FN void pbr_eval(const ShadingParams& sp, Vec3 wo, Vec3 wi, Spectrum& f) {
  f.n = sp.albedo.n;
  if (wo.z <= 0.f || wi.z <= 0.f) {
    f.fill(0.f);
    return;
  }
  Vec3 h = normalize(wo + wi);
  GGX ggx{pbr_alpha(sp)};
  float voh = maxf(0.f, dot(wo, h));
  float spec = ggx.D(h) * ggx.G(wo, wi) / (4.f * wo.z * wi.z);
  float f0d = sqr((sp.ior - 1.f) / (sp.ior + 1.f));
  float fd = clampf(sp.specular, 0.f, 1.f) * schlick(f0d, voh);
  float met = clampf(sp.metallic, 0.f, 1.f);
  float diel_spec = (1.f - met) * fd * spec;
  float diff = (1.f - met) * (1.f - fd) * kInvPi;
  for (int i = 0; i < f.n; ++i) {
    float a = sp.albedo.v[i];
    f.v[i] = diff * a + diel_spec + met * schlick(a, voh) * spec;
  }
}

SPECTRAL_FN float pbr_pdf(const ShadingParams& sp, Vec3 wo, Vec3 wi) {
  if (wo.z <= 0.f || wi.z <= 0.f) return 0.f;
  Vec3 h = normalize(wo + wi);
  GGX ggx{pbr_alpha(sp)};
  float ps = pbr_spec_prob(sp);
  float voh = dot(wo, h);
  float spec_pdf = voh > 0.f ? ggx.pdf_visible(wo, h) / (4.f * voh) : 0.f;
  return ps * spec_pdf + (1.f - ps) * wi.z * kInvPi;
}

SPECTRAL_FN bool pbr_sample(const ShadingParams& sp, Vec3 wo, float u_lobe, Vec2 u, Vec3* wi) {
  if (wo.z <= 0.f) return false;
  float ps = pbr_spec_prob(sp);
  if (u_lobe < ps) {
    GGX ggx{pbr_alpha(sp)};
    Vec3 h = ggx.sample_visible(wo, u);
    *wi = reflect(wo, h);
  } else {
    *wi = sample_cosine_hemisphere(u);
  }
  return wi->z > 0.f;
}

// ---------------------------------------------------------------- dielectrics
SPECTRAL_FN float band_ior(const ShadingParams& sp, int i) { return sp.ior_bands ? sp.ior_bands[i] : sp.ior; }

// Thin slab reflectance/transmittance for one band given the single-interface Fresnel F
// and internal transmittance tau (one pass): incoherent multiple reflections.
SPECTRAL_FN void thin_slab_rt(float F, float tau, float* R, float* T) {
  float denom = 1.f - F * F * tau * tau;
  denom = denom > 1e-6f ? denom : 1e-6f;
  *R = F + (1.f - F) * (1.f - F) * F * tau * tau / denom;
  *T = (1.f - F) * (1.f - F) * tau / denom;
}

// Per-band internal transmittance along a refracted path through a thin slab.
// tau0 is specified for normal incidence; path length scales with 1/cos_t.
SPECTRAL_FN float thin_tau(const ShadingParams& sp, int i, float cos_i) {
  if (!sp.trans_bands) return 1.f;
  float eta = sp.ior;
  float sin2_t = (1.f - cos_i * cos_i) / (eta * eta);
  float cos_t = safe_sqrt(1.f - minf(sin2_t, 0.999999f));
  float t0 = clampf(sp.trans_bands[i], 0.f, 1.f);
  return t0 > 0.f ? powf(t0, 1.f / maxf(cos_t, 1e-3f)) : 0.f;
}

// Samples a delta event and multiplies the per-band weight f*cos/pdf into beta.
// `n` is the shading normal (world), `front` whether wo is on the outside of the surface.
SPECTRAL_FN bool glass_sample(const ShadingParams& sp, Vec3 wo, Vec3 n, bool front, float u, Spectrum& beta,
                              BsdfSample& bs) {
  Vec3 nn = dot(n, wo) >= 0.f ? n : -n;  // normal on the side of wo
  float cos_i = dot(nn, wo);
  if (cos_i <= 0.f) return false;
  const int nb = beta.n;
  if (sp.type == kMatThinDielectric) {
    float rsum = 0.f, tsum = 0.f;
    for (int i = 0; i < nb; ++i) {
      float R, T;
      thin_slab_rt(fresnel_dielectric(cos_i, band_ior(sp, i)), thin_tau(sp, i, cos_i), &R, &T);
      rsum += R;
      tsum += T;
    }
    if (rsum + tsum <= 0.f) return false;
    float pr = rsum / (rsum + tsum);
    bool reflect_event = u < pr;
    for (int i = 0; i < nb; ++i) {
      float R, T;
      thin_slab_rt(fresnel_dielectric(cos_i, band_ior(sp, i)), thin_tau(sp, i, cos_i), &R, &T);
      beta.v[i] *= reflect_event ? R / pr : T / (1.f - pr);
    }
    bs.pdf = 1.f;
    if (reflect_event) {
      bs.wi = reflect(wo, nn);
      bs.event = kEventSpecularReflect;
    } else {
      bs.wi = -wo;
      bs.event = kEventPassThrough;
    }
    return true;
  }
  // Solid dielectric interface. eta = n_t / n_i.
  float eta_ref = front ? sp.ior : 1.f / sp.ior;
  Vec3 wt;
  bool can_refract = refract_dir(wo, nn, eta_ref, &wt);
  float fsum = 0.f;
  for (int i = 0; i < nb; ++i) {
    float e = front ? band_ior(sp, i) : 1.f / band_ior(sp, i);
    fsum += can_refract ? fresnel_dielectric(cos_i, e) : 1.f;
  }
  float pr = fsum / float(nb);
  bool reflect_event = !can_refract || u < pr;
  float radiance_scale = 1.f / (eta_ref * eta_ref);
  for (int i = 0; i < nb; ++i) {
    float e = front ? band_ior(sp, i) : 1.f / band_ior(sp, i);
    float F = can_refract ? fresnel_dielectric(cos_i, e) : 1.f;
    if (reflect_event)
      beta.v[i] *= can_refract ? F / pr : 1.f;
    else
      beta.v[i] *= (1.f - F) / (1.f - pr) * radiance_scale;
  }
  bs.pdf = 1.f;
  if (reflect_event) {
    bs.wi = reflect(wo, nn);
    bs.event = kEventSpecularReflect;
  } else {
    bs.wi = normalize(wt);
    bs.event = kEventSpecularTransmit;
  }
  return true;
}

// Transmittance of a glass surface for a straight shadow ray (transparent shadows).
// dir: shadow ray direction; entering: ray enters the object at this surface.
SPECTRAL_FN void glass_shadow_transmittance(const ShadingParams& sp, Vec3 n, Vec3 dir, bool entering, Spectrum& T) {
  float cos_i = fabsf(dot(n, dir));
  for (int i = 0; i < T.n; ++i) {
    if (sp.type == kMatThinDielectric) {
      float R, Tt;
      thin_slab_rt(fresnel_dielectric(cos_i, band_ior(sp, i)), thin_tau(sp, i, cos_i), &R, &Tt);
      T.v[i] *= Tt;
    } else {
      float e = entering ? band_ior(sp, i) : 1.f / band_ior(sp, i);
      float F = fresnel_dielectric(cos_i, e);
      T.v[i] *= (1.f - F);
    }
  }
}

}  // namespace spectral
