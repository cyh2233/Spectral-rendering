// Macbeth ColorChecker under D65: rendered spectra -> CIE XYZ -> CIELAB vs. reference
// computed directly from the measured reflectances (no rendering).
#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/io/csv_spectrum.h"
#include "spectral/spectrum/colorimetry.h"
#include "spectral/util/paths.h"

using namespace spectral;
using nlohmann::json;

namespace {
// Pixel whose 5x5 neighbourhood lies on the same patch (no mixing with the background).
bool interior(const SpectralImage& img, int x, int y, uint32_t id) {
  for (int dy = -2; dy <= 2; ++dy)
    for (int dx = -2; dx <= 2; ++dx) {
      int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= img.width || yy >= img.height) return false;
      if (img.seg_id[size_t(yy) * img.width + xx] != id) return false;
    }
  return true;
}

struct CheckerResult {
  std::vector<double> de;
};

CheckerResult render_checker(bool use_uplift) {
  auto cc = read_csv_spectra(data_dir() + "/spectra/colorchecker/colorchecker_babel.csv");
  const DenseSpectrum d65 = DenseSpectrum::cie_illuminant("D65");
  const auto white_xyz = d65.xyz();

  json insts = json::array();
  const int cols = 6, rows = 4;
  for (int i = 0; i < 24; ++i) {
    int cx = i % cols, cy = i / cols;
    json mat = {{"specular", 0.0}};
    if (use_uplift) {
      auto xyz = (cc.spectrum(i) * d65).xyz();
      Vec3d rgb = xyz_to_linear_srgb({xyz[0] / white_xyz[1], xyz[1] / white_xyz[1], xyz[2] / white_xyz[1]});
      mat["base_color"] = {std::max(0.0, rgb[0]), std::max(0.0, rgb[1]), std::max(0.0, rgb[2])};
    } else {
      mat["reflectance"] = "colorchecker_babel:" + cc.column_names[i];
    }
    insts.push_back({{"primitive", "quad"},
                     {"size", {1.0, 1.0}},
                     {"translate", {(cx - 2.5) * 1.2, 0.0, (cy - 1.5) * 1.2}},
                     {"seg_id", i + 1},
                     {"material", mat}});
  }
  json j = {{"render", {{"spp", 1}, {"max_depth", 1}, {"bands", {{"min", 380}, {"max", 780}, {"step", 5}}}}},
            {"camera", {{"position", {0, 50, 0}}, {"look_at", {0, 0, 0}}, {"up", {0, 0, -1}}, {"fov_deg", 9.0},
                        {"resolution", {160, 110}}}},
            {"instances", insts},
            {"lights", {{{"type", "directional"}, {"direction", {0, -1, 0}},
                         {"emission", {{"type", "cie"}, {"name", "D65"}, {"scale", 0.01}}}}}}};
  SpectralImage img = test::render_json(j);
  std::vector<float> w = band_cmf_weights(img.grid);
  // Perfect white reflector under the same light: L = E / pi.
  std::vector<float> e_bands = (d65 * (0.01f / kPi)).to_bands(img.grid);
  Vec3d white_r = bands_to_xyz(e_bands.data(), img.grid, w);

  CheckerResult res;
  for (int i = 0; i < 24; ++i) {
    // Average all pixels of this patch (via segmentation AOV).
    std::vector<double> acc(img.grid.n, 0.0);
    int n = 0;
    for (int y = 0; y < img.height; ++y)
      for (int x = 0; x < img.width; ++x)
        if (interior(img, x, y, uint32_t(i + 1))) {
          const float* p = img.pixel(x, y);
          for (int b = 0; b < img.grid.n; ++b) acc[b] += p[b];
          ++n;
        }
    REQUIRE(n > 20);
    std::vector<float> mean(img.grid.n);
    for (int b = 0; b < img.grid.n; ++b) mean[b] = float(acc[b] / n);
    Vec3d xyz = bands_to_xyz(mean.data(), img.grid, w);
    Vec3d lab = xyz_to_lab(xyz, white_r);
    auto ref = (cc.spectrum(i) * d65).xyz();
    auto refw = d65.xyz();
    Vec3d lab_ref = xyz_to_lab({ref[0], ref[1], ref[2]}, {refw[0], refw[1], refw[2]});
    res.de.push_back(delta_e76(lab, lab_ref));
  }
  return res;
}
}  // namespace

TEST_CASE("ColorChecker under D65 with measured spectra: dE76 < 0.5") {
  CheckerResult r = render_checker(false);
  double mean = 0, mx = 0;
  for (double d : r.de) {
    mean += d / 24;
    mx = std::max(mx, d);
  }
  MESSAGE("measured spectra: mean dE76 = " << mean << ", max = " << mx);
  CHECK(mx < 0.5);
  CHECK(mean < 0.15);
}

TEST_CASE("ColorChecker from sRGB base colours via uplifting: small colour error") {
  CheckerResult r = render_checker(true);
  double mean = 0, mx = 0;
  for (double d : r.de) {
    mean += d / 24;
    mx = std::max(mx, d);
  }
  MESSAGE("uplifted sRGB: mean dE76 = " << mean << ", max = " << mx);
  CHECK(mean < 0.6);
  CHECK(mx < 6.0);  // out-of-gamut patches (e.g. cyan) are clipped in sRGB
}
