#include <initializer_list>

#include "doctest.h"
#include "spectral/kernel/bsdf.h"
#include "spectral/kernel/rng.h"

using namespace spectral;

namespace {
ShadingParams pbr(float metallic, float roughness, float albedo, float specular = 1.f) {
  ShadingParams sp;
  sp.albedo = Spectrum(8, albedo);
  sp.metallic = metallic;
  sp.roughness = roughness;
  sp.specular = specular;
  return sp;
}
}  // namespace

TEST_CASE("PBR BSDF is reciprocal") {
  Rng rng;
  for (float met : {0.f, 0.5f, 1.f})
    for (float rough : {0.1f, 0.5f, 1.f}) {
      ShadingParams sp = pbr(met, rough, 0.7f);
      for (int i = 0; i < 200; ++i) {
        Vec3 a = sample_cosine_hemisphere(rng.next2d()), b = sample_cosine_hemisphere(rng.next2d());
        Spectrum fab, fba;
        pbr_eval(sp, a, b, fab);
        pbr_eval(sp, b, a, fba);
        CHECK(fab.v[3] == doctest::Approx(fba.v[3]).epsilon(1e-4));
      }
    }
}

TEST_CASE("PBR BSDF: importance sampling is consistent with pdf() and conserves energy") {
  for (float met : {0.f, 1.f})
    for (float rough : {0.2f, 0.6f}) {
      ShadingParams sp = pbr(met, rough, 1.f);
      Vec3 wo = normalize(Vec3(0.3f, 0.2f, 0.9f));
      Rng rng = Rng::for_pixel_sample(1, 2, 3, 4);
      double est_is = 0, est_uniform = 0;
      const int N = 300000;
      for (int i = 0; i < N; ++i) {
        Vec3 wi;
        if (pbr_sample(sp, wo, rng.next1d(), rng.next2d(), &wi)) {
          float pdf = pbr_pdf(sp, wo, wi);
          Spectrum f;
          pbr_eval(sp, wo, wi, f);
          if (pdf > 0) est_is += f.v[0] * wi.z / pdf;
        }
        Vec3 wu = sample_cosine_hemisphere(rng.next2d());
        Spectrum f;
        pbr_eval(sp, wo, wu, f);
        est_uniform += f.v[0] * wu.z / cosine_hemisphere_pdf(wu.z);
      }
      est_is /= N;
      est_uniform /= N;
      CHECK(est_is == doctest::Approx(est_uniform).epsilon(0.03));
      CHECK(est_is <= 1.01);
    }
  // Pure Lambert (specular 0) integrates exactly to the albedo.
  ShadingParams lam = pbr(0.f, 1.f, 0.6f, 0.f);
  Vec3 wo(0, 0, 1);
  Rng rng;
  double acc = 0;
  for (int i = 0; i < 10000; ++i) {
    Vec3 wi;
    REQUIRE(pbr_sample(lam, wo, rng.next1d(), rng.next2d(), &wi));
    Spectrum f;
    pbr_eval(lam, wo, wi, f);
    acc += f.v[0] * wi.z / pbr_pdf(lam, wo, wi);
  }
  CHECK(acc / 10000 == doctest::Approx(0.6).epsilon(1e-4));
}

TEST_CASE("Fresnel: normal incidence and total internal reflection") {
  CHECK(fresnel_dielectric(1.f, 1.5f) == doctest::Approx(0.04f).epsilon(1e-4));
  CHECK(fresnel_dielectric(1.f, 1.f / 1.5f) == doctest::Approx(0.04f).epsilon(1e-4));
  CHECK(fresnel_dielectric(0.2f, 1.f / 1.5f) == 1.f);
  float R, T;
  thin_slab_rt(0.04f, 1.f, &R, &T);
  CHECK(R + T == doctest::Approx(1.f).epsilon(1e-6));
  CHECK(T == doctest::Approx(0.96f * 0.96f / (1 - 0.04f * 0.04f)).epsilon(1e-6));
}

TEST_CASE("smooth dielectric sampling preserves expected energy per band") {
  ShadingParams sp;
  sp.type = kMatDielectric;
  sp.ior = 1.5f;
  float bands[4] = {1.52f, 1.51f, 1.50f, 1.49f};
  sp.ior_bands = bands;
  Vec3 n(0, 0, 1);
  Vec3 wo = normalize(Vec3(0.5f, 0.f, 0.8f));
  Rng rng;
  double refl[4] = {}, trans[4] = {};
  const int N = 200000;
  for (int i = 0; i < N; ++i) {
    Spectrum beta(4, 1.f);
    BsdfSample bs;
    REQUIRE(glass_sample(sp, wo, n, true, rng.next1d(), beta, bs));
    for (int b = 0; b < 4; ++b) (bs.event == kEventSpecularReflect ? refl : trans)[b] += beta.v[b];
  }
  for (int b = 0; b < 4; ++b) {
    float F = fresnel_dielectric(wo.z, bands[b]);
    CHECK(refl[b] / N == doctest::Approx(F).epsilon(0.03));
    // Transmitted radiance is scaled by 1/eta^2 (radiance compression).
    CHECK(trans[b] / N == doctest::Approx((1 - F) / (1.5f * 1.5f)).epsilon(0.01));
  }
}
