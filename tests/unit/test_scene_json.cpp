#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/kernel/camera.h"
#include "spectral/kernel/light.h"
#include "spectral/scene/gltf_loader.h"
#include <cstdlib>

#include "spectral/sky/sky_model.h"

using namespace spectral;
using nlohmann::json;

TEST_CASE("scene JSON: camera from look_at / focal length") {
  json j = {{"camera", {{"position", {1, 2, 3}}, {"look_at", {1, 2, 0}}, {"resolution", {200, 100}},
                        {"focal_length_mm", 18.0}, {"sensor_width_mm", 36.0}}}};
  auto s = load_scene_json(j, ".");
  CHECK(s->camera.width == 200);
  CHECK(s->camera.tan_half_fov_x == doctest::Approx(1.0f));
  CHECK(s->camera.tan_half_fov_y == doctest::Approx(0.5f));
  Ray r = generate_camera_ray(s->camera, 100.f, 50.f);
  CHECK(r.o.z == doctest::Approx(3.f));
  CHECK(r.d.z == doctest::Approx(-1.f));
  Ray right = generate_camera_ray(s->camera, 200.f, 50.f);
  CHECK(right.d.x > 0.5f);  // +X is to the right when looking down -Z
}

TEST_CASE("scene JSON: errors name the offending key") {
  json bad = {{"instances", {{{"primitive", "torus"}}}}};
  CHECK_THROWS_WITH_AS(load_scene_json(bad, "."), doctest::Contains("instances[0].primitive"), std::runtime_error);
  json bad2 = {{"materials", {{"definitions", {{"x", {{"reflectance", "no_such_spectrum"}}}}}}}};
  CHECK_THROWS_WITH_AS(load_scene_json(bad2, "."), doctest::Contains("no_such_spectrum"), std::runtime_error);
  json bad3 = {{"camera", {{"fov_axis", "sideways"}}}};
  CHECK_THROWS_AS(load_scene_json(bad3, "."), std::runtime_error);
}

TEST_CASE("scene JSON: photometric normalization and glass IOR spectra") {
  json j = {{"materials",
             {{"definitions",
               {{"lamp", {{"emission", {{"type", "blackbody"}, {"temperature", 3000}, {"photometric", 1000.0}}}}},
                {"windshield", {{"bsdf", "thin_dielectric"}, {"glass", "soda_lime"}}}}}}}};
  auto s = load_scene_json(j, ".");
  const auto& mats = s->materials();
  REQUIRE(mats.size() == 2);
  auto le = s->band_spectrum(mats[0].emission_spec);
  DenseSpectrum d = from_bands(le, s->grid);
  CHECK(683.0 * d.xyz()[1] == doctest::Approx(1000.0).epsilon(2e-2));
  CHECK(mats[1].type == kMatThinDielectric);
  CHECK(mats[1].ior == doctest::Approx(1.52f).epsilon(5e-3));
  auto n = s->band_spectrum(mats[1].ior_spec);
  CHECK(n.front() > n.back());  // normal dispersion
}

TEST_CASE("glTF loader: hierarchy, materials, textures, transmission") {
  Scene scene;
  GltfAsset a = load_gltf(test::data_path("gltf/test_asset.gltf"), scene);
  REQUIRE(a.parts.size() == 3);
  CHECK(scene.materials().size() == 2);
  CHECK(scene.materials()[1].type == kMatThinDielectric);
  CHECK(scene.materials()[1].ior == doctest::Approx(1.52f));
  REQUIRE(scene.textures().size() == 1);
  CHECK(scene.textures()[0].width == 2);
  CHECK(scene.textures()[0].srgb);
  CHECK(scene.textures()[0].wrap == kWrapClamp);
  // panel_b: root translation (0,1,0) then (2,0,0) + 90 deg about Y: local +X -> world -Z.
  const GltfPart& pb = a.parts[1];
  CHECK(pb.node_name == "panel_b");
  Vec3 p = pb.node_to_asset.point(Vec3(1, 0, 0));
  CHECK(p.x == doctest::Approx(2.f).epsilon(1e-5));
  CHECK(p.y == doctest::Approx(1.f).epsilon(1e-5));
  CHECK(p.z == doctest::Approx(-1.f).epsilon(1e-5));
}

TEST_CASE("scene JSON: asset instances, overrides and segmentation by material") {
  json j = {{"assets", {{"obj", {{"path", "gltf/test_asset.gltf"}}}}},
            {"materials", {{"overrides", {{{"match", "Paint"}, {"roughness", 0.2}}}}}},
            {"instances",
             {{{"asset", "obj"}, {"seg_id", 5}, {"seg_id_by_material", {{"Glass", 9}}}},
              {{"asset", "obj"},
               {"translate", {10, 0, 0}},
               {"material_overrides", {{{"match", "Paint"}, {"reflectance", 0.3}}}}}}}};
  auto s = load_scene_json(j, SPECTRAL_TEST_DATA_DIR);
  REQUIRE(s->instances().size() == 6);
  CHECK(s->instances()[0].seg_id == 5);
  CHECK(s->instances()[2].seg_id == 9);  // glass part
  const auto& m0 = s->materials()[s->instances()[0].material];
  CHECK(m0.roughness == doctest::Approx(0.2f));
  const auto& m3 = s->materials()[s->instances()[3].material];
  CHECK(m3.reflectance_spec >= 0);
  CHECK(m3.roughness == doctest::Approx(0.2f));  // global override kept in the clone
  CHECK(s->instances()[3].to_world.m[0][3] == doctest::Approx(10.f));
  CHECK(s->params.has_glass == 1);
}

TEST_CASE("sky plugin through the C ABI") {
  json j = {{"render", {{"bands", {{"min", 400}, {"max", 700}, {"step", 10}}}}},
            {"lights",
             {{{"type", "sky"}, {"model", "plugin"}, {"library", SPECTRAL_TEST_SKY_PLUGIN},
               {"sun", {{"elevation_deg", 30}, {"azimuth_deg", 90}}}, {"table_resolution", {16, 8}}, {"cache", false}}}}};
  auto s = load_scene_json(j, ".");
  SceneView v = s->view();
  REQUIRE(v.env.present);
  Spectrum L;
  env_lookup(v, Vec3(0, 1, 0), L);
  CHECK(L.v[0] == doctest::Approx(0.5f));
  env_lookup(v, Vec3(0, -1, 0), L);
  CHECK(L.v[0] == 0.f);
  REQUIRE(v.sun_light >= 0);
  Vec3 sd = s->lights()[v.sun_light].direction;
  CHECK(sd.y == doctest::Approx(0.5f).epsilon(1e-4));
  CHECK(sd.x == doctest::Approx(std::cos(30 * kPi / 180)).epsilon(1e-4));  // azimuth 90 = east = +X
  CHECK(sun_lookup(v, sd, L));
  CHECK(L.v[0] == doctest::Approx(1000.f * 400.f / 500.f).epsilon(1e-3));
}

TEST_CASE("default material overrides map CARLA-style names to spectra") {
  json j = {{"assets", {{"obj", {{"path", "gltf/test_asset.gltf"}}}}},
            {"materials", {{"use_default_overrides", true}}},
            {"instances", {{{"asset", "obj"}}}}};
  auto s = load_scene_json(j, SPECTRAL_TEST_DATA_DIR);
  const auto& glass = s->materials()[s->instances()[2].material];
  CHECK(glass.type == kMatThinDielectric);  // "M_Glass" matches (?i)(window|glass)
  CHECK(glass.ior_spec >= 0);
  const auto& paint = s->materials()[s->instances()[0].material];
  CHECK(paint.type == kMatPbr);
  CHECK(paint.reflectance_spec < 0);  // no rule for "M_Paint_Red"
  CHECK_THROWS_WITH_AS(load_scene_json({{"materials", {{"overrides", {{{"match", "("}}}}}},
                                        {"assets", {{"obj", {{"path", "gltf/test_asset.gltf"}}}}},
                                        {"instances", {{{"asset", "obj"}}}}},
                                       SPECTRAL_TEST_DATA_DIR),
                       doctest::Contains("invalid regular expression"), std::runtime_error);
}

TEST_CASE("Prague sky model (runs only when SPECTRAL_PRAGUE_DATASET is set)") {
  const char* ds = std::getenv("SPECTRAL_PRAGUE_DATASET");
  if (!ds || !prague_sky_available()) {
    MESSAGE("skipped: set SPECTRAL_PRAGUE_DATASET to a PragueSkyModelDataset*.dat file");
    return;
  }
  json j = {{"render", {{"bands", {{"min", 400}, {"max", 1000}, {"step", 20}}}}},
            {"lights", {{{"type", "sky"}, {"model", "prague"}, {"dataset", ds},
                         {"sun", {{"elevation_deg", 45}, {"azimuth_deg", 180}}}, {"table_resolution", {64, 32}},
                         {"cache", false}}}}};
  auto s = load_scene_json(j, ".");
  SceneView v = s->view();
  Spectrum zen, hor;
  env_lookup(v, Vec3(0, 1, 0), zen);
  env_lookup(v, normalize(Vec3(1, 0.05f, 0)), hor);
  CHECK(zen.v[s->grid.nearest_band(450.f)] > zen.v[s->grid.nearest_band(700.f)]);  // blue sky
  REQUIRE(v.sun_light >= 0);
  Spectrum sun;
  CHECK(sun_lookup(v, s->lights()[v.sun_light].direction, sun));
  // Direct normal irradiance at 550 nm for a clear sky at 45 deg: order 1 W/(m^2 nm).
  double half = std::acos(s->lights()[v.sun_light].cos_max);
  double e550 = sun.v[s->grid.nearest_band(560.f)] * 2 * kPi * (1 - std::cos(half));
  CHECK(e550 > 0.5);
  CHECK(e550 < 2.0);
}
