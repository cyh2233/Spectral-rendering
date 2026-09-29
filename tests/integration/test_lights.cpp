// Light transport against analytic references: area lights (NEE + MIS), sun disk, spot cone,
// importance-sampled HDR environment maps, textured glTF assets.
#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/spectrum/colorimetry.h"
#include "tinyexr.h"

using namespace spectral;
using nlohmann::json;

namespace {
json floor_scene(int spp, int max_depth) {
  json j = {{"render", {{"spp", spp}, {"max_depth", max_depth}, {"seed", 5},
                        {"bands", {{"min", 400}, {"max", 1000}, {"step", 20}}}}},
            {"camera", {{"position", {0, 0.5, 0}}, {"look_at", {0, 0, 0}}, {"up", {0, 0, -1}}, {"fov_deg", 2},
                        {"resolution", {6, 6}}}}};
  j["instances"] = json::array(
      {{{"primitive", "quad"}, {"size", {200, 200}}, {"material", {{"reflectance", 0.5}, {"specular", 0.0}}}}});
  return j;
}
}  // namespace

TEST_CASE("spherical area light: L = rho * Le * (r/d)^2 below the sphere (NEE + MIS)") {
  const double r = 0.5, d = 3.0, Le = 4.0;
  json j = floor_scene(256, 1);
  j["instances"].push_back({{"primitive", "sphere"}, {"radius", r}, {"segments", 256}, {"rings", 128},
                            {"translate", {0, d, 0}},
                            {"material", {{"reflectance", 0.0}, {"emission", Le}}}});
  // The camera sits below the light; move it sideways so it does not look through the sphere.
  SpectralImage img = test::render_json(j);
  double expect = 0.5 * Le * (r / d) * (r / d);
  CHECK(test::mean_all(test::mean_bands(img, 0, 0, 6, 6)) == doctest::Approx(expect).epsilon(0.01));
}

TEST_CASE("sun disk: irradiance pi L sin^2(alpha) cos(theta) on a Lambertian floor") {
  const double half = 0.5 * kPi / 180.0, el = 50.0, Ls = 2000.0;
  json j = floor_scene(64, 1);
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"},
                              {"sun", {{"elevation_deg", el}, {"azimuth_deg", 10}, {"half_angle_deg", 0.5},
                                       {"radiance", Ls}}}}});
  SpectralImage img = test::render_json(j);
  double theta = (90.0 - el) * kPi / 180.0;
  double E = kPi * Ls * std::sin(half) * std::sin(half) * std::cos(theta);
  CHECK(test::mean_all(test::mean_bands(img, 0, 0, 6, 6)) == doctest::Approx(0.5 * E / kPi).epsilon(0.01));
}

TEST_CASE("spot light: full intensity inside the inner cone, zero outside the outer cone") {
  json j = floor_scene(4, 1);
  j["lights"] = json::array({{{"type", "spot"}, {"position", {0, 2, 0}}, {"direction", {0, -1, 0}},
                              {"cone_inner_deg", 10}, {"cone_outer_deg", 20}, {"emission", 8.0}}});
  SpectralImage img = test::render_json(j);
  CHECK(test::mean_all(test::mean_bands(img, 0, 0, 6, 6)) == doctest::Approx(0.5 * 8.0 / (kPi * 4.0)).epsilon(1e-3));
  j["camera"]["position"] = {1.5, 0.5, 0};  // looks at (1.5, 0, 0): 36.9 deg off-axis
  j["camera"]["look_at"] = {1.5, 0, 0};
  SpectralImage off = test::render_json(j);
  CHECK(test::mean_all(test::mean_bands(off, 0, 0, 6, 6)) == doctest::Approx(0.0));
}

TEST_CASE("HDR environment map: importance sampling reproduces the cosine-weighted integral") {
  // 64x32 gray lat-long map with a bright hotspot; gray RGB uplifts to c * D65n exactly.
  const int W = 64, H = 32;
  std::vector<float> rgb(size_t(W) * H * 3);
  auto value = [&](int x, int y) {
    float c = 0.2f;
    if (x >= 20 && x < 24 && y >= 6 && y < 9) c = 60.f;
    return c;
  };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int k = 0; k < 3; ++k) rgb[(size_t(y) * W + x) * 3 + k] = value(x, y);
  std::string path = test::tmp_path("env_hotspot.exr");
  const char* err = nullptr;
  REQUIRE(SaveEXR(rgb.data(), W, H, 3, 0, path.c_str(), &err) == TINYEXR_SUCCESS);

  json j = floor_scene(512, 1);
  j["lights"] = json::array({{{"type", "envmap"}, {"path", path}, {"resolution", {W, H}}}});
  SpectralImage img = test::render_json(j);

  // Reference: (rho/pi) * sum over texels of c * integral(cos theta dOmega) over the upper hemisphere.
  double integral = 0;
  const int sub = 16;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int sy = 0; sy < sub; ++sy) {
        double theta = kPi * (y + (sy + 0.5) / sub) / H;
        double cos_t = std::cos(theta);
        if (cos_t <= 0) continue;
        double domega = (2 * kPi / W) * (kPi / H / sub) * std::sin(theta);
        integral += value(x, y) * cos_t * domega;
      }
  Scene tmp;
  tmp.set_band_grid(400, 1000, 20);
  auto d65n = tmp.d65n_bands();
  auto m = test::mean_bands(img, 0, 0, 6, 6);
  for (int b = 0; b < img.grid.n; b += 5)
    CHECK(m[b] == doctest::Approx(0.5 / kPi * integral * d65n[b]).epsilon(0.02));
}

TEST_CASE("textured glTF asset renders the expected colours") {
  json j = {{"render", {{"spp", 64}, {"max_depth", 1}, {"bands", {{"min", 380}, {"max", 780}, {"step", 10}}}}},
            {"camera", {{"position", {0, 1, 2}}, {"look_at", {0, 1, 0}}, {"fov_deg", 30}, {"resolution", {40, 40}}}}};
  j["assets"]["obj"]["path"] = "gltf/test_asset.gltf";
  j["materials"]["overrides"] = json::array();
  j["materials"]["overrides"].push_back({{"match", "(?i)glass"}, {"bsdf", "thin_dielectric"}, {"transmittance", 1.0}});
  j["materials"]["overrides"].push_back({{"match", "(?i)PAINT"}, {"specular", 0.0}});
  j["instances"] = json::array();
  j["instances"].push_back({{"asset", "obj"}});
  j["lights"] = json::array();
  j["lights"].push_back({{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 1.0}});
  SpectralImage img = test::render_json(j);
  auto w = band_cmf_weights(img.grid);
  auto rgb_at = [&](int x0, int y0) {  // mean over a 4x4 block
    std::vector<double> m = test::mean_bands(img, x0, y0, x0 + 4, y0 + 4);
    std::vector<float> px(m.begin(), m.end());
    return xyz_to_linear_srgb(bands_to_xyz(px.data(), img.grid, w));
  };
  // Panel spans pixels ~[1, 39]; texel-pure regions are within ~9 px of each corner.
  Vec3d tl = rgb_at(3, 3), tr = rgb_at(33, 3), bl = rgb_at(3, 33), br = rgb_at(33, 33);
  CHECK(tl[0] > 2 * tl[1]);
  CHECK(tl[0] > 2 * tl[2]);  // red
  CHECK(tr[1] > 2 * tr[0]);
  CHECK(tr[1] > 2 * tr[2]);  // green
  CHECK(bl[2] > 2 * bl[0]);
  CHECK(bl[2] > 2 * bl[1]);  // blue
  // White texel under an equal-energy sky has the chromaticity of illuminant E (not D65 white).
  std::vector<float> ones(img.grid.n, 1.f);
  Vec3d e = xyz_to_linear_srgb(bands_to_xyz(ones.data(), img.grid, w));
  CHECK(br[0] / br[1] == doctest::Approx(e[0] / e[1]).epsilon(0.05));
  CHECK(br[2] / br[1] == doctest::Approx(e[2] / e[1]).epsilon(0.05));
  CHECK(img.seg_id[20 * 40 + 20] == 0u);
}
