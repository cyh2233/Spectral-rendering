// Analytic validation of the dense-band path tracer.
#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/io/csv_spectrum.h"
#include "spectral/util/paths.h"

using namespace spectral;
using nlohmann::json;

namespace {
json base_scene(int w, int h, int spp) {
  return json{{"render", {{"spp", spp}, {"seed", 7}, {"bands", {{"min", 380}, {"max", 1000}, {"step", 5}}}}},
              {"camera", {{"resolution", {w, h}}}}};
}
}  // namespace

TEST_CASE("closed emissive sphere furnace: L = Le / (1 - rho) per band") {
  json j = base_scene(8, 8, 64);
  j["camera"]["position"] = {0, 0, 0};
  j["camera"]["look_at"] = {0, 0, -1};
  j["camera"]["fov_deg"] = 90;
  j["render"]["max_depth"] = 400;
  j["render"]["rr_depth"] = 5;
  json refl = {{"type", "samples"}, {"wavelengths", {380, 1000}}, {"values", {0.2, 0.8}}};
  j["instances"] = json::array({{{"primitive", "sphere"},
                                 {"radius", 10.0},
                                 {"segments", 32},
                                 {"rings", 16},
                                 {"material",
                                  {{"reflectance", refl}, {"specular", 0.0}, {"emission", 1.0}, {"double_sided", true}}}}});
  SpectralImage img = test::render_json(j);
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  int bad = 0;
  for (int b = 0; b < img.grid.n; ++b) {
    double rho = 0.2 + 0.6 * (img.grid.center(b) - 380.0) / 620.0;
    double expect = 1.0 / (1.0 - rho);
    if (std::fabs(m[b] / expect - 1.0) > 0.03) ++bad;
  }
  CHECK(bad == 0);
  CHECK(m[0] == doctest::Approx(1.25).epsilon(0.03));
  CHECK(m[img.grid.n - 1] == doctest::Approx(5.0).epsilon(0.03));
}

TEST_CASE("white furnace: convex Lambertian sphere under a uniform sky") {
  json j = base_scene(24, 24, 32);
  j["camera"]["position"] = {0, 0, 4};
  j["camera"]["look_at"] = {0, 0, 0};
  j["camera"]["fov_deg"] = 40;
  j["instances"] = json::array({{{"primitive", "sphere"},
                                 {"radius", 1.0},
                                 {"material", {{"reflectance", 0.8}, {"specular", 0.0}}}}});
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 1.0}}});
  SpectralImage img = test::render_json(j);
  auto centre = test::mean_bands(img, 9, 9, 15, 15);
  auto corner = test::mean_bands(img, 0, 0, 2, 2);
  CHECK(test::mean_all(centre) == doctest::Approx(0.8).epsilon(0.01));
  CHECK(test::mean_all(corner) == doctest::Approx(1.0).epsilon(1e-4));
  for (int b = 0; b < img.grid.n; b += 10) CHECK(centre[b] == doctest::Approx(0.8).epsilon(0.03));
}

TEST_CASE("Lambertian plane under a directional light: L = rho E cos / pi (exact, per band)") {
  json j = base_scene(8, 8, 4);
  j["camera"]["position"] = {0, 2, 0};
  j["camera"]["look_at"] = {0, 0, 0};
  j["camera"]["up"] = {0, 0, -1};
  j["camera"]["fov_deg"] = 10;
  j["render"]["max_depth"] = 1;
  j["instances"] = json::array({{{"primitive", "quad"},
                                 {"size", {100, 100}},
                                 {"material", {{"reflectance", "colorchecker_babel:orange"}, {"specular", 0.0}}}}});
  const float th = 30.f * kPi / 180.f;
  j["lights"] = json::array({{{"type", "directional"},
                              {"direction", {std::sin(th), -std::cos(th), 0.0}},
                              {"emission", {{"type", "cie"}, {"name", "A"}, {"scale", 0.01}}}}});
  SpectralImage img = test::render_json(j);
  auto cc = read_csv_spectra(data_dir() + "/spectra/colorchecker/colorchecker_babel.csv");
  auto rho = cc.spectrum(cc.column_index("orange")).to_bands(img.grid);
  auto E = (DenseSpectrum::cie_illuminant("A") * 0.01f).to_bands(img.grid);
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  for (int b = 0; b < img.grid.n; ++b) {
    double expect = rho[b] * E[b] * std::cos(th) / kPi;
    CHECK(m[b] == doctest::Approx(expect).epsilon(1e-4));
  }
}

TEST_CASE("point light inverse-square law and AOVs") {
  json j = base_scene(9, 9, 4);
  j["camera"]["position"] = {0, 2, 0};
  j["camera"]["look_at"] = {0, 0, 0};
  j["camera"]["up"] = {0, 0, -1};
  j["camera"]["fov_deg"] = 1;
  j["render"]["max_depth"] = 1;
  j["instances"] = json::array({{{"primitive", "quad"},
                                 {"size", {10, 10}},
                                 {"seg_id", 42},
                                 {"material", {{"reflectance", 0.5}, {"specular", 0.0}}}}});
  j["lights"] = json::array({{{"type", "point"}, {"position", {0, 3, 0}}, {"emission", 10.0}}});
  SpectralImage img = test::render_json(j);
  auto m = test::mean_bands(img, 4, 4, 5, 5);
  CHECK(test::mean_all(m) == doctest::Approx(0.5 * 10.0 / (kPi * 9.0)).epsilon(1e-3));
  CHECK(img.depth[4 * 9 + 4] == doctest::Approx(2.f).epsilon(1e-5));
  CHECK(img.seg_id[4 * 9 + 4] == 42u);
}

TEST_CASE("rendering is deterministic across thread counts") {
  json j = base_scene(16, 12, 8);
  j["camera"]["position"] = {0, 0, 4};
  j["camera"]["look_at"] = {0, 0, 0};
  j["instances"] = json::array({{{"primitive", "sphere"}, {"material", {{"reflectance", 0.6}}}},
                                {{"primitive", "quad"}, {"size", {20, 20}}, {"translate", {0, -1, 0}}}});
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 0.3},
                              {"sun", {{"elevation_deg", 40}, {"azimuth_deg", 30}, {"irradiance", 1.0}}}}});
  SpectralImage a = test::render_json(j, 1), b = test::render_json(j, 4);
  CHECK(a.radiance == b.radiance);
  CHECK(a.depth == b.depth);
  CHECK(a.seg_id == b.seg_id);
}

TEST_CASE("sky pixels have infinite depth and the sky segmentation id") {
  json j = base_scene(4, 4, 1);
  j["camera"]["position"] = {0, 0, 0};
  j["camera"]["look_at"] = {0, 1, 0};
  j["camera"]["up"] = {0, 0, -1};
  SpectralImage img = test::render_json(j);
  CHECK(std::isinf(img.depth[5]) == false);  // stored as FLT_MAX
  CHECK(img.depth[5] > 1e30f);
  CHECK(img.seg_id[5] == kSkySegId);
}
