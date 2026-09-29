#include <initializer_list>

#include "doctest.h"
#include "spectral/kernel/math.h"
#include "spectral/kernel/rng.h"
#include "spectral/kernel/sampling.h"

using namespace spectral;

TEST_CASE("affine inverse and composition") {
  Affine a = Affine::identity();
  a.m[0][0] = 2.f; a.m[0][1] = 0.5f; a.m[1][2] = -1.f; a.m[2][2] = 3.f;
  a.m[0][3] = 1.f; a.m[1][3] = -2.f; a.m[2][3] = 5.f;
  Affine ai = inverse(a);
  Vec3 p(0.3f, -1.7f, 2.2f);
  Vec3 q = ai.point(a.point(p));
  CHECK(q.x == doctest::Approx(p.x).epsilon(1e-5));
  CHECK(q.y == doctest::Approx(p.y).epsilon(1e-5));
  CHECK(q.z == doctest::Approx(p.z).epsilon(1e-5));
  Affine c = compose(a, ai);
  for (int r = 0; r < 3; ++r)
    for (int k = 0; k < 4; ++k) CHECK(c.m[r][k] == doctest::Approx(r == k ? 1.f : 0.f).epsilon(1e-5));
}

TEST_CASE("frame is orthonormal and right handed") {
  Rng rng;
  for (int i = 0; i < 1000; ++i) {
    Vec3 n = sample_uniform_sphere(rng.next2d());
    Frame f = Frame::from_z(n);
    CHECK(dot(f.x, f.y) == doctest::Approx(0.f).epsilon(1e-5));
    CHECK(length(f.x) == doctest::Approx(1.f).epsilon(1e-5));
    Vec3 c = cross(f.x, f.y);
    CHECK(dot(c, f.z) == doctest::Approx(1.f).epsilon(1e-4));
  }
}

TEST_CASE("pcg32 determinism and uniformity") {
  Rng a = Rng::for_pixel_sample(7, 10, 20, 3), b = Rng::for_pixel_sample(7, 10, 20, 3);
  for (int i = 0; i < 100; ++i) CHECK(a.next_u32() == b.next_u32());
  Rng c = Rng::for_pixel_sample(7, 10, 20, 4);
  Rng d = Rng::for_pixel_sample(7, 10, 20, 3);
  CHECK(c.next_u32() != d.next_u32());
  // Chi-square on 16 bins.
  Rng r = Rng::for_pixel_sample(1, 2, 3, 4);
  const int N = 160000, B = 16;
  int bins[B] = {};
  for (int i = 0; i < N; ++i) {
    float u = r.next1d();
    REQUIRE(u >= 0.f);
    REQUIRE(u < 1.f);
    bins[int(u * B)]++;
  }
  double chi2 = 0, e = double(N) / B;
  for (int i = 0; i < B; ++i) chi2 += (bins[i] - e) * (bins[i] - e) / e;
  CHECK(chi2 < 40.0);  // 15 dof, p ~ 5e-4
}

TEST_CASE("cosine hemisphere sampling integrates to one") {
  Rng r;
  double acc = 0;
  const int N = 200000;
  for (int i = 0; i < N; ++i) {
    Vec3 w = sample_cosine_hemisphere(r.next2d());
    acc += (w.z / kPi) / cosine_hemisphere_pdf(w.z);
  }
  CHECK(acc / N == doctest::Approx(1.0).epsilon(1e-3));
}

TEST_CASE("GGX VNDF pdf integrates to one over reflected directions") {
  // integral of pdf(wi) over the sphere must be <= 1 (== 1 minus directions below horizon).
  for (float alpha : {0.1f, 0.4f, 0.9f}) {
    GGX g{alpha};
    Vec3 wo = normalize(Vec3(0.4f, 0.1f, 0.8f));
    Rng r;
    double acc = 0;
    const int N = 400000;
    for (int i = 0; i < N; ++i) {
      Vec3 wi = sample_uniform_sphere(r.next2d());
      Vec3 h = normalize(wo + wi);
      float voh = dot(wo, h);
      if (voh <= 0.f || h.z <= 0.f) continue;
      float pdf = g.pdf_visible(wo, h) / (4.f * voh);
      acc += pdf / uniform_sphere_pdf();
    }
    CHECK(acc / N == doctest::Approx(1.0).epsilon(0.03));
  }
}

