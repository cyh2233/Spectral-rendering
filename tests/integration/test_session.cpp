// Live-session behaviour: incremental updates must give the same image as a scene built from
// scratch in the final state.
#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/session.h"
#include "spectral/sky/sky_model.h"

using namespace spectral;
using nlohmann::json;

namespace {
json base() {
  json j = {{"render", {{"spp", 4}, {"seed", 3}, {"max_depth", 3}, {"bands", {{"min", 400}, {"max", 1000}, {"step", 20}}}}},
            {"camera", {{"position", {0, 1.5, 6}}, {"look_at", {0, 0.5, 0}}, {"fov_deg", 50}, {"resolution", {48, 32}}}}};
  j["instances"] = json::array({{{"primitive", "quad"}, {"size", {30, 30}}, {"seg_id", 1},
                                 {"material", {{"reflectance", 0.4}, {"specular", 0.0}}}}});
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 0.2},
                              {"sun", {{"elevation_deg", 40}, {"azimuth_deg", 30}, {"irradiance", 2.0}}}}});
  return j;
}
json box(float x) {
  return {{"primitive", "box"}, {"size", {1, 1, 1}}, {"translate", {x, 0.5, 0}}, {"seg_id", 7},
          {"material", {{"reflectance", "approx_car_paint_red"}}}};
}
Affine translation(float x, float y, float z) {
  Affine a = Affine::identity();
  a.m[0][3] = x;
  a.m[1][3] = y;
  a.m[2][3] = z;
  return a;
}
SpectralImage render_session(Session& s, int spp) {
  s.reset();
  s.render(spp);
  return s.image();
}
}  // namespace

TEST_CASE("session: moved instance renders identically to a fresh scene") {
  Session live("cpu", 2);
  live.load_json(base().dump(), SPECTRAL_TEST_DATA_DIR);
  int h = live.spawn(box(-2.f).dump());
  live.render(1);  // prepare with the initial pose
  live.set_transform(h, translation(1.5f, 0.5f, 0.f));
  SpectralImage a = render_session(live, 4);

  json fresh = base();
  fresh["instances"].push_back(box(1.5f));
  Session ref("cpu", 2);
  ref.load_json(fresh.dump(), SPECTRAL_TEST_DATA_DIR);
  SpectralImage b = render_session(ref, 4);
  CHECK(a.radiance == b.radiance);
  CHECK(a.seg_id == b.seg_id);
}

TEST_CASE("session: remove hides the object and handles are recycled") {
  Session live("cpu", 2);
  live.load_json(base().dump(), SPECTRAL_TEST_DATA_DIR);
  int h = live.spawn(box(0.f).dump());
  SpectralImage with = render_session(live, 2);
  live.remove(h);
  SpectralImage without = render_session(live, 2);
  Session ref("cpu", 2);
  ref.load_json(base().dump(), SPECTRAL_TEST_DATA_DIR);
  SpectralImage b = render_session(ref, 2);
  REQUIRE(without.radiance.size() == b.radiance.size());
  double maxdiff = 0;
  for (size_t i = 0; i < b.radiance.size(); ++i)
    maxdiff = std::max(maxdiff, double(std::fabs(without.radiance[i] - b.radiance[i])));
  CHECK(maxdiff < 1e-6);
  CHECK(std::count(with.seg_id.begin(), with.seg_id.end(), 7u) > 10);
  CHECK(std::count(without.seg_id.begin(), without.seg_id.end(), 7u) == 0);
  size_t n_inst = live.scene().instances().size();
  int h2 = live.spawn(box(1.f).dump());
  CHECK(h2 == h);
  CHECK(live.scene().instances().size() == n_inst);
  CHECK(live.live_objects() == 1);
}

TEST_CASE("session: dynamic lights and sun motion") {
  json j = base();
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 0.0},
                              {"sun", {{"elevation_deg", 90}, {"azimuth_deg", 0}, {"irradiance", 3.0}}}}});
  j["render"]["max_depth"] = 1;
  j["camera"] = {{"position", {0, 1, 0}}, {"look_at", {0, 0, 0}}, {"up", {0, 0, -1}}, {"fov_deg", 2}, {"resolution", {4, 4}}};
  Session s("cpu", 1);
  s.load_json(j.dump(), SPECTRAL_TEST_DATA_DIR);
  auto mean = [](const SpectralImage& im) { return test::mean_all(test::mean_bands(im, 0, 0, im.width, im.height)); };
  double noon = mean(render_session(s, 8));
  CHECK(noon == doctest::Approx(0.4 * 3.0 / kPi).epsilon(0.02));
  s.set_sun(sun_direction_from_angles(30.0, 0.0));
  CHECK(mean(render_session(s, 8)) == doctest::Approx(noon * 0.5).epsilon(0.02));
  s.set_sun(sun_direction_from_angles(-10.0, 0.0));  // night: the sun is kept with zero radiance
  CHECK(mean(render_session(s, 4)) == doctest::Approx(0.0));
  s.set_dynamic_lights(json::array({{{"type", "point"}, {"position", {0, 2, 0}}, {"emission", 4.0}}}).dump());
  CHECK(mean(render_session(s, 4)) == doctest::Approx(0.4 * 4.0 / (kPi * 4.0)).epsilon(1e-3));
  size_t pool = s.scene().spectra().size();
  s.set_dynamic_lights(json::array({{{"type", "point"}, {"position", {0, 3, 0}}, {"emission", 4.0}}}).dump());
  CHECK(s.scene().spectra().size() == pool);  // identical emission specs share one pooled spectrum
  s.set_dynamic_lights("[]");
  CHECK(mean(render_session(s, 4)) == doctest::Approx(0.0));
}

TEST_CASE("session: preview modes") {
  json j = base();
  j["lights"] = json::array({{{"type", "sky"}, {"model", "simple"}, {"sky_radiance", 0.5}}});
  j["camera"] = {{"position", {0, 0, 0}}, {"look_at", {0, 1, 0}}, {"up", {0, 0, -1}}, {"fov_deg", 10}, {"resolution", {8, 6}}};
  Session s("cpu", 1);
  s.load_json(j.dump(), SPECTRAL_TEST_DATA_DIR);
  s.render(2);
  PreviewOptions o;
  o.mode = "band";
  o.auto_exposure = false;
  auto rgb = s.preview(o);
  REQUIRE(rgb.size() == 8 * 6 * 3);
  CHECK(int(rgb[0]) == int(std::lround(preview_encode(0.5f) * 255.f)));
  o.mode = "seg";
  rgb = s.preview(o);
  CHECK(int(rgb[0]) == 0);  // sky
  o.mode = "srgb";
  o.auto_exposure = true;
  rgb = s.preview(o);
  CHECK(rgb[0] > 100);  // auto exposure brings the sky to mid grey
  o.mode = "false_color";
  CHECK_NOTHROW(s.preview(o));
  o.mode = "nope";
  CHECK_THROWS(s.preview(o));
}
