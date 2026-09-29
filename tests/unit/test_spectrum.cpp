#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/io/csv_spectrum.h"
#include "spectral/spectrum/colorimetry.h"
#include "spectral/spectrum/dense_spectrum.h"
#include "spectral/spectrum/glass.h"
#include "spectral/util/paths.h"

using namespace spectral;

TEST_CASE("dense spectrum interpolation and extrapolation") {
  auto s = DenseSpectrum::from_samples({400.f, 500.f}, {0.2f, 0.6f}, Extrapolation::Clamp);
  CHECK(s.eval(450.f) == doctest::Approx(0.4f));
  CHECK(s.eval(350.f) == doctest::Approx(0.2f));
  CHECK(s.eval(900.f) == doctest::Approx(0.6f));
  auto z = DenseSpectrum::from_samples({400.f, 500.f}, {0.2f, 0.6f}, Extrapolation::Zero);
  CHECK(z.eval(350.f) == 0.f);
  CHECK(z.eval(900.f) == 0.f);
  CHECK_THROWS(DenseSpectrum::from_samples({500.f, 400.f}, {1.f, 1.f}));
}

TEST_CASE("band box-average of a linear ramp equals its centre value") {
  auto ramp = DenseSpectrum::from_samples({300.f, 1100.f}, {0.f, 800.f});
  BandGrid g{380.f, 5.f, 125};
  auto b = ramp.to_bands(g);
  for (int i = 0; i < g.n; ++i) CHECK(b[i] == doctest::Approx(g.center(i) - 300.f).epsilon(1e-4));
}

TEST_CASE("CIE D65 white point and illuminant A chromaticity") {
  auto d65 = DenseSpectrum::cie_illuminant("D65").xyz();
  CHECK(d65[0] / d65[1] == doctest::Approx(0.95047).epsilon(1e-3));
  CHECK(d65[2] / d65[1] == doctest::Approx(1.08883).epsilon(1e-3));
  auto a = DenseSpectrum::cie_illuminant("A").xyz();
  double s = a[0] + a[1] + a[2];
  CHECK(a[0] / s == doctest::Approx(0.44757).epsilon(5e-4));
  CHECK(a[1] / s == doctest::Approx(0.40745).epsilon(5e-4));
  CHECK(DenseSpectrum::cie_illuminant("A").eval(560.f) == doctest::Approx(100.f).epsilon(1e-4));
  // Normalized D65 has the luminance of a constant 1 spectrum.
  CHECK(d65_normalized().xyz()[1] == doctest::Approx(DenseSpectrum::constant(1.f).xyz()[1]).epsilon(1e-4));
}

TEST_CASE("blackbody: Wien peak and absolute radiance") {
  auto bb = DenseSpectrum::blackbody(5000.f);
  int peak = 0;
  for (int i = 0; i < DenseSpectrum::kSamples; ++i)
    if (bb.at_index(i) > bb.at_index(peak)) peak = i;
  CHECK(DenseSpectrum::lambda_at(peak) == doctest::Approx(2.897771955e6 / 5000.0).epsilon(2e-3));
  // Planck at 500 nm, 5000 K: 1.2107e13 W/(m^2 sr m) -> 12107 W/(m^2 sr nm).
  CHECK(bb.eval(500.f) == doctest::Approx(12107.f).epsilon(2e-3));
}

TEST_CASE("CSV spectra: ColorChecker columns and emitters") {
  auto cc = read_csv_spectra(data_dir() + "/spectra/colorchecker/colorchecker_babel.csv");
  CHECK(cc.columns.size() == 24);
  CHECK(cc.column_index("dark_skin") == 0);
  CHECK(cc.column_index("white_9_5_05_d") == 18);
  CHECK(cc.extrapolation == Extrapolation::Clamp);
  auto white = cc.spectrum(18);
  CHECK(white.eval(550.f) > 0.85f);
  auto hps = read_csv_spectra(data_dir() + "/spectra/emitters/cie_hp1.csv");
  CHECK(hps.extrapolation == Extrapolation::Zero);
  CHECK(hps.spectrum(0).eval(900.f) == 0.f);
}

TEST_CASE("glass catalog: N-BK7 matches Schott n_d and dispersion") {
  auto bk7 = find_glass("N-BK7");
  CHECK(bk7.ior(587.56) == doctest::Approx(1.5168).epsilon(1e-4));
  double nF = bk7.ior(486.13), nC = bk7.ior(656.27);
  double abbe = (bk7.ior(587.56) - 1.0) / (nF - nC);
  CHECK(abbe == doctest::Approx(64.17).epsilon(3e-3));
  auto sl = find_glass("soda_lime");
  CHECK(sl.ior(589.0) == doctest::Approx(1.52).epsilon(5e-3));
}

TEST_CASE("colorimetry: sRGB <-> XYZ and Lab") {
  Vec3d white = linear_srgb_to_xyz({1, 1, 1});
  CHECK(white[1] == doctest::Approx(1.0).epsilon(1e-4));
  Vec3d rgb = xyz_to_linear_srgb(white);
  for (double c : rgb) CHECK(c == doctest::Approx(1.0).epsilon(1e-4));
  Vec3d lab = xyz_to_lab(white, white);
  CHECK(lab[0] == doctest::Approx(100.0));
  CHECK(srgb_decode(srgb_encode(0.3f)) == doctest::Approx(0.3f).epsilon(1e-5));
}
