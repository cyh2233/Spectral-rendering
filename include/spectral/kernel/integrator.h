// Dense-band spectral path tracer (unidirectional, NEE + MIS, Russian roulette).
//
// Each path carries throughput and radiance for ALL output bands. Sampling decisions
// (lobe choice, directions, light selection) are wavelength independent, so there is
// no hero-wavelength machinery. Dispersion (wavelength-dependent directions) is not
// simulated in the renderer; per-band Fresnel is.
//
// The template parameter `Isect` must provide:
//   bool closest(const Ray&, float tmax, Hit*) const;   // alpha-masked closest hit
//   bool occluded(const Ray&, float tmax) const;        // any opaque hit
#pragma once
#include "spectral/kernel/bsdf.h"
#include "spectral/kernel/camera.h"
#include "spectral/kernel/light.h"
#include "spectral/kernel/rng.h"

namespace spectral {

struct PathAov {
  float depth = kInf;
  uint32_t seg_id = kSkySegId;
};

SPECTRAL_FN Ray spawn_ray(Vec3 p, Vec3 ng, Vec3 dir) {
  Ray r;
  r.o = offset_ray_origin(p, dot(ng, dir) >= 0.f ? ng : -ng);
  r.d = dir;
  return r;
}

// Visibility with transparent glass: returns false if blocked, else T holds the per-band
// transmittance (Fresnel + absorption of glass surfaces crossed).
template <class Isect>
SPECTRAL_FN bool shadow_transmittance(const SceneView& s, const Isect& isect, Vec3 p, Vec3 ng, Vec3 dir,
                                      float dist, Spectrum& T) {
  T.n = s.grid.n;
  T.fill(1.f);
  Ray r = spawn_ray(p, ng, dir);
  float remaining = dist < kInf ? dist * (1.f - 1e-4f) : kInf;
  if (!s.params.has_glass || !s.params.transparent_shadows) return !isect.occluded(r, remaining);
  for (int layer = 0; layer < 16; ++layer) {
    Hit h;
    if (!isect.closest(r, remaining, &h)) return true;
    const InstanceRecord& inst = s.instances[h.instance];
    if (inst.material < 0 || !is_glass(s.materials[inst.material])) return false;
    SurfaceInteraction si = make_interaction(s, h.instance, h.prim, h.b1, h.b2, r.d, h.t);
    const MaterialRecord& m = s.materials[inst.material];
    ShadingParams sp;
    evaluate_material(s, si, sp);
    if (m.type == kMatDielectric && !si.front && m.absorption_spec >= 0) {
      const float* sig = s.spectra + m.absorption_spec;
      for (int i = 0; i < T.n; ++i) T.v[i] *= expf(-sig[i] * h.t);
    }
    glass_shadow_transmittance(sp, sp.frame.z, r.d, si.front, T);
    if (T.max_value() <= 0.f) return false;
    r = spawn_ray(si.p, si.ng, r.d);
    if (remaining < kInf) remaining -= h.t;
  }
  return false;
}

// Straight-line probe from the camera that skips glass: depth / segmentation AOVs of
// the first opaque surface (so a windshield does not hide the scene in the AOVs).
template <class Isect>
SPECTRAL_FN PathAov probe_aov(const SceneView& s, const Isect& isect, Ray r) {
  PathAov a;
  Vec3 fwd = camera_forward(s.camera);
  float travelled = 0.f;
  for (int layer = 0; layer < 16; ++layer) {
    Hit h;
    if (!isect.closest(r, kInf, &h)) return a;
    const InstanceRecord& inst = s.instances[h.instance];
    travelled += h.t;
    if (inst.material >= 0 && is_glass(s.materials[inst.material])) {
      SurfaceInteraction si = make_interaction(s, h.instance, h.prim, h.b1, h.b2, r.d, h.t);
      r = spawn_ray(si.p, si.ng, r.d);
      continue;
    }
    a.depth = s.params.depth_mode == kDepthZ ? travelled * dot(r.d, fwd) : travelled;
    a.seg_id = inst.seg_id;
    return a;
  }
  return a;
}

template <class Isect>
SPECTRAL_FN void trace_path(const SceneView& s, const Isect& isect, Ray ray, Rng& rng, Spectrum& L) {
  const int nb = s.grid.n;
  L.n = nb;
  L.fill(0.f);
  Spectrum beta(nb, 1.f);
  Spectrum tmp, tmp2;
  tmp.n = nb;
  tmp2.n = nb;

  // MIS state of the last non-specular vertex.
  float prev_pdf = 0.f;
  bool prev_specular = true;  // camera vertex acts as a delta
  Vec3 prev_p = ray.o;

  int glass_events = 0;
  for (int bounce = 0;;) {
    Hit hit;
    if (!isect.closest(ray, kInf, &hit)) {
      // Escaped: environment + sun disk.
      if (s.env.present) {
        env_lookup(s, ray.d, tmp);
        float w = 1.f;
        if (!prev_specular && s.env_light >= 0)
          w = power_heuristic(prev_pdf, light_pmf(s, s.env_light) * env_pdf(s.env, ray.d));
        L.add_product(beta, tmp, w);
      }
      if (sun_lookup(s, ray.d, tmp)) {
        const LightRecord& sl = s.lights[s.sun_light];
        float w = prev_specular ? 1.f : power_heuristic(prev_pdf, light_pmf(s, s.sun_light) * uniform_cone_pdf(sl.cos_max));
        L.add_product(beta, tmp, w);
      }
      break;
    }

    SurfaceInteraction si = make_interaction(s, hit.instance, hit.prim, hit.b1, hit.b2, ray.d, hit.t);
    if (si.material < 0) break;
    const MaterialRecord& mat = s.materials[si.material];

    // Absorption inside a solid dielectric (segment from the entry point to this exit point).
    if (mat.type == kMatDielectric && !si.front && mat.absorption_spec >= 0) {
      const float* sig = s.spectra + mat.absorption_spec;
      for (int i = 0; i < nb; ++i) beta.v[i] *= expf(-sig[i] * hit.t);
    }

    // Emission.
    if (material_emits(mat) && (mat.double_sided || si.front)) {
      evaluate_emission(s, mat, si.uv, tmp);
      float w = 1.f;
      const InstanceRecord& inst = s.instances[si.instance];
      if (!prev_specular && inst.light_offset >= 0) {
        int32_t li = inst.light_offset + int32_t(si.prim);
        float lp = light_pmf(s, li) * triangle_light_pdf(s.lights[li], prev_p, si.p, si.ng);
        w = power_heuristic(prev_pdf, lp);
      }
      L.add_product(beta, tmp, w);
    }

    if (bounce >= s.params.max_depth) break;

    ShadingParams sp;
    evaluate_material(s, si, sp);

    BsdfSample bs;
    if (mat.type != kMatPbr) {
      // Glass: delta events only. They do not count towards max_depth (a windshield in front
      // of the camera must not consume the bounce budget), but are bounded separately.
      if (++glass_events > s.params.max_glass_events) break;
      if (!glass_sample(sp, si.wo, sp.frame.z, si.front, rng.next1d(), beta, bs)) break;
      (void)rng.next2d();  // keep dimension usage aligned with the PBR branch
      bool through = (bs.event == kEventPassThrough || bs.event == kEventSpecularTransmit);
      if (!(through && s.params.transparent_shadows)) {
        prev_specular = true;
      }
      // With transparent shadows, transmission keeps the MIS state of the last
      // non-specular vertex (NEE already accounted for light through glass).
      ray = spawn_ray(si.p, si.ng, bs.wi);
    } else {
      // Orient the shading frame towards wo (two-sided shading).
      Frame fr = sp.frame;
      if (dot(fr.z, si.wo) < 0.f) {
        fr.z = -fr.z;
        fr.x = -fr.x;
      }
      Vec3 wo_l = fr.to_local(si.wo);
      if (wo_l.z <= 0.f) break;

      // Next-event estimation.
      if (s.n_lights > 0) {
        float pmf;
        int32_t li = int32_t(sample_alias(s.light_alias, s.n_lights, rng.next1d(), &pmf));
        LightSample ls;
        Vec2 ul = rng.next2d();
        if (pmf > 0.f && sample_light(s, li, si.p, ul, ls, tmp)) {
          Vec3 wi_l = fr.to_local(ls.wi);
          // Reject light directions below the geometric surface on the wo side.
          bool geo_ok = dot(ls.wi, si.ng) * dot(si.wo, si.ng) > 0.f;
          if (wi_l.z > 0.f && geo_ok && !tmp.is_black()) {
            pbr_eval(sp, wo_l, wi_l, tmp2);
            float light_pdf = ls.pdf * pmf;
            float w = ls.is_delta ? 1.f : power_heuristic(light_pdf, pbr_pdf(sp, wo_l, wi_l));
            Spectrum vis;
            if (shadow_transmittance(s, isect, si.p, si.ng, ls.wi, ls.dist, vis)) {
              float k = w * wi_l.z / light_pdf;
              for (int i = 0; i < nb; ++i) L.v[i] += beta.v[i] * tmp2.v[i] * tmp.v[i] * vis.v[i] * k;
            }
          }
        }
      } else {
        (void)rng.next1d();
        (void)rng.next2d();
      }

      // BSDF sampling.
      Vec3 wi_l;
      float u_lobe = rng.next1d();
      Vec2 u2 = rng.next2d();
      if (!pbr_sample(sp, wo_l, u_lobe, u2, &wi_l)) break;
      Vec3 wi = fr.to_world(wi_l);
      if (dot(wi, si.ng) * dot(si.wo, si.ng) <= 0.f) break;  // light leak guard
      float pdf = pbr_pdf(sp, wo_l, wi_l);
      if (!(pdf > 0.f)) break;
      pbr_eval(sp, wo_l, wi_l, tmp2);
      float k = wi_l.z / pdf;
      for (int i = 0; i < nb; ++i) beta.v[i] *= tmp2.v[i] * k;
      prev_pdf = pdf;
      prev_specular = false;
      prev_p = si.p;
      ray = spawn_ray(si.p, si.ng, wi);
      ++bounce;
    }

    // Russian roulette.
    float m = beta.max_value();
    if (!(m > 0.f)) break;
    if (bounce + glass_events >= s.params.rr_depth) {
      float q = minf(1.f, m);
      if (rng.next1d() >= q) break;
      beta *= 1.f / q;
    }
  }

  if (s.params.clamp_contribution > 0.f) {
    float m = L.max_value();
    if (m > s.params.clamp_contribution) L *= s.params.clamp_contribution / m;
  }
}

// One camera sample for pixel (px, py).
template <class Isect>
SPECTRAL_FN void render_sample(const SceneView& s, const Isect& isect, int px, int py, uint32_t sample, Spectrum& L,
                               PathAov* aov) {
  Rng rng = Rng::for_pixel_sample(s.params.seed, uint32_t(px), uint32_t(py), sample);
  Vec2 j = rng.next2d();
  Ray ray = generate_camera_ray(s.camera, float(px) + j.x, float(py) + j.y);
  if (aov) {
    Ray center = generate_camera_ray(s.camera, float(px) + 0.5f, float(py) + 0.5f);
    *aov = probe_aov(s, isect, center);
  }
  trace_path(s, isect, ray, rng, L);
}

}  // namespace spectral
