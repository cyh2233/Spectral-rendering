#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/kernel/rng.h"
#include "spectral/kernel/uplift.h"
#include "spectral/spectrum/colorimetry.h"
#include "spectral/spectrum/uplift_lut.h"

using namespace spectral;

namespace {
// Reflectance spectrum (dense) from coefficients, evaluated without the NIR hold.
DenseSpectrum coeff_spectrum(const UpliftCoeffs& c) {
  DenseSpectrum s;
  for (int i = 0; i < DenseSpectrum::kSamples; ++i) s.at_index(i) = uplift_eval(c, DenseSpectrum::lambda_at(i));
  return s;
}
Vec3d reflectance_to_srgb(const DenseSpectrum& r) {
  const DenseSpectrum& d65 = DenseSpectrum::cie_illuminant("D65");
  auto xyz = (r * d65).xyz();
  auto w = d65.xyz();
  return xyz_to_linear_srgb({xyz[0] / w[1], xyz[1] / w[1], xyz[2] / w[1]});
}
}  // namespace

TEST_CASE("Jakob-Hanika uplifting round trip (sRGB -> spectrum -> sRGB)") {
  auto lut = obtain_uplift_lut();
  REQUIRE(lut->res >= 16);
  UpliftLutView v = lut->view();
  Rng rng = Rng::for_pixel_sample(11, 0, 0, 0);
  double max_err = 0, sum_err = 0;
  const int N = 500;
  for (int i = 0; i < N; ++i) {
    float rgb[3] = {0.05f + 0.9f * rng.next1d(), 0.05f + 0.9f * rng.next1d(), 0.05f + 0.9f * rng.next1d()};
    Vec3d back = reflectance_to_srgb(coeff_spectrum(uplift_fetch(v, rgb[0], rgb[1], rgb[2])));
    Vec3d w = linear_srgb_to_xyz({1, 1, 1});
    double de = delta_e76(xyz_to_lab(linear_srgb_to_xyz({rgb[0], rgb[1], rgb[2]}), w), xyz_to_lab(linear_srgb_to_xyz(back), w));
    max_err = std::max(max_err, de);
    sum_err += de;
  }
  MESSAGE("uplift round trip: mean dE76 = " << sum_err / N << ", max = " << max_err);
  CHECK(sum_err / N < 0.25);
  CHECK(max_err < 1.0);
  // Gray closed form and bounds.
  UpliftCoeffs g = uplift_fetch(v, 0.18f, 0.18f, 0.18f);
  CHECK(uplift_eval(g, 550.f) == doctest::Approx(0.18f).epsilon(1e-5));
  CHECK(uplift_eval(uplift_fetch(v, 1.f, 1.f, 1.f), 700.f) == doctest::Approx(1.f));
  CHECK(uplift_eval(uplift_fetch(v, 0.f, 0.f, 0.f), 700.f) == doctest::Approx(0.f));
}

TEST_CASE("uplifted reflectance is held constant beyond the NIR hold wavelength") {
  Scene s;
  s.uplift = obtain_uplift_lut();
  s.params.uplift_hold_lambda = 780.f;
  SceneView v = s.view();
  Spectrum sp;
  uplift_reflectance(v, 0.2f, 0.6f, 0.3f, sp);
  int b780 = s.grid.nearest_band(780.f);
  for (int b = b780; b < s.grid.n; ++b) CHECK(sp.v[b] == doctest::Approx(sp.v[b780]).epsilon(1e-6));
  CHECK(sp.v[s.grid.nearest_band(550.f)] > sp.v[s.grid.nearest_band(450.f)]);  // greenish
}

TEST_CASE("RGB illuminant uplifting: white has the luminance of a constant 1 spectrum") {
  Scene s;
  s.uplift = obtain_uplift_lut();
  SceneView v = s.view();
  Spectrum sp;
  uplift_illuminant(v, 2.f, 2.f, 2.f, sp);
  std::vector<float> w = band_cmf_weights(s.grid);
  Vec3d xyz = bands_to_xyz(sp.v, s.grid, w);
  std::vector<float> ones(s.grid.n, 2.f);
  Vec3d ref = bands_to_xyz(ones.data(), s.grid, w);
  CHECK(xyz[1] == doctest::Approx(ref[1]).epsilon(5e-3));
}
