// Windshield-relevant glass transport: thin panes, solid slabs, transparent shadows.
#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/kernel/bsdf.h"
#include "spectral/spectrum/glass.h"

using namespace spectral;
using nlohmann::json;

namespace {
json view_through_pane(json pane_material, int spp, int max_depth) {
  // Camera at z=+2 looking at -Z; pane (normal +Z) at z=0; emissive backdrop at z=-2 facing +Z.
  json j = {{"render", {{"spp", spp}, {"max_depth", max_depth}, {"seed", 3}, {"rr_depth", 50},
                        {"bands", {{"min", 400}, {"max", 1000}, {"step", 20}}}}},
            {"camera", {{"position", {0, 0, 2}}, {"look_at", {0, 0, 0}}, {"fov_deg", 5}, {"resolution", {10, 10}}}}};
  j["instances"] = json::array({
      {{"primitive", "quad"}, {"size", {4, 4}}, {"rotate_deg", {90, 0, 0}}, {"material", pane_material}},
      {{"primitive", "quad"}, {"size", {40, 40}}, {"rotate_deg", {90, 0, 0}}, {"translate", {0, 0, -2}},
       {"material", {{"reflectance", 0.0}, {"emission", 1.0}}}}});
  return j;
}
}  // namespace

TEST_CASE("thin pane transmittance at normal incidence (per band, Sellmeier IOR)") {
  // rotate_deg 90 about X maps the quad normal +Y to +Z.
  json pane = {{"bsdf", "thin_dielectric"}, {"glass", "soda_lime"}, {"transmittance", 0.8}};
  SpectralImage img = test::render_json(view_through_pane(pane, 256, 8));
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  GlassModel g = find_glass("soda_lime");
  for (int b = 0; b < img.grid.n; ++b) {
    float n = from_bands(g.spectrum().to_bands(img.grid), img.grid).eval(img.grid.center(b));
    float F = fresnel_dielectric(1.f, n), R, T;
    thin_slab_rt(F, 0.8f, &R, &T);
    CHECK(m[b] == doctest::Approx(T).epsilon(0.01));
  }
}

TEST_CASE("solid slab with absorption: T = (1-F)^2 e^{-a d} / (1 - F^2 e^{-2 a d})") {
  const double d = 0.01, sigma = 20.0;  // 1 cm, 20 / m
  json j = view_through_pane({}, 512, 64);
  j["instances"][0] = {{"primitive", "box"}, {"size", {4, 4, d}},
                       {"material", {{"bsdf", "dielectric"}, {"ior", 1.5}, {"absorption", sigma}}}};
  SpectralImage img = test::render_json(j);
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  double F = fresnel_dielectric(1.f, 1.5f), tau = std::exp(-sigma * d);
  double T = (1 - F) * (1 - F) * tau / (1 - F * F * tau * tau);
  CHECK(test::mean_all(m) == doctest::Approx(T).epsilon(0.01));
}

TEST_CASE("pane reflectance (ghost/veiling glare path) at normal incidence") {
  json pane = {{"bsdf", "thin_dielectric"}, {"ior", 1.5}};
  json j = view_through_pane(pane, 256, 8);
  // Replace backdrop by a black one, add an emitter behind the camera facing the pane.
  j["instances"][1]["material"] = {{"reflectance", 0.0}};
  j["instances"].push_back({{"primitive", "quad"}, {"size", {40, 40}}, {"rotate_deg", {-90, 0, 0}},
                            {"translate", {0, 0, 3}}, {"material", {{"reflectance", 0.0}, {"emission", 1.0}}}});
  SpectralImage img = test::render_json(j);
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  float R, T;
  thin_slab_rt(fresnel_dielectric(1.f, 1.5f), 1.f, &R, &T);
  CHECK(test::mean_all(m) == doctest::Approx(R).epsilon(0.03));
}

TEST_CASE("transparent shadows: sunlight through a thin pane onto a Lambertian floor (exact)") {
  const float th = 20.f * kPi / 180.f;
  json j = {{"render", {{"spp", 2}, {"max_depth", 1}, {"bands", {{"min", 400}, {"max", 1000}, {"step", 20}}}}},
            {"camera", {{"position", {0, 1, 0}}, {"look_at", {0, 0, 0}}, {"up", {0, 0, -1}}, {"fov_deg", 2},
                        {"resolution", {4, 4}}}}};
  j["instances"] = json::array({
      {{"primitive", "quad"}, {"size", {100, 100}}, {"material", {{"reflectance", 0.5}, {"specular", 0.0}}}},
      // Horizontal pane above the floor, offset so the camera ray does not cross it.
      {{"primitive", "quad"}, {"size", {4, 4}}, {"translate", {3.0, 2.0, 0.0}},
       {"material", {{"bsdf", "thin_dielectric"}, {"ior", 1.5}, {"transmittance", 0.9}}}}});
  // Light travels towards -Y tilted to -X: rays reaching the floor origin cross the pane at x ~ 0.73.
  j["instances"][1]["translate"] = {0.73, 2.0, 0.0};
  j["lights"] = json::array({{{"type", "directional"}, {"direction", {-std::sin(th), -std::cos(th), 0.0}},
                              {"emission", 1.0}}});
  SpectralImage img = test::render_json(j);
  float F = fresnel_dielectric(std::cos(th), 1.5f), R, T;
  ShadingParams sp;
  sp.type = kMatThinDielectric;
  sp.ior = 1.5f;
  float tau[1] = {0.9f};
  sp.trans_bands = tau;
  thin_slab_rt(F, thin_tau(sp, 0, std::cos(th)), &R, &T);
  auto m = test::mean_bands(img, 0, 0, img.width, img.height);
  CHECK(test::mean_all(m) == doctest::Approx(0.5 * std::cos(th) * T / kPi).epsilon(1e-4));
  // Without transparent shadows the pane blocks the light completely.
  j["render"]["transparent_shadows"] = false;
  SpectralImage blocked = test::render_json(j);
  CHECK(test::mean_all(test::mean_bands(blocked, 0, 0, 4, 4)) == doctest::Approx(0.0));
}
