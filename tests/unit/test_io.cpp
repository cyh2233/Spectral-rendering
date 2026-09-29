#include "doctest.h"
#include "helpers/test_util.h"
#include "spectral/io/image_io.h"

using namespace spectral;

namespace {
SpectralImage random_image(int w, int h, BandGrid g) {
  SpectralImage img;
  img.width = w;
  img.height = h;
  img.grid = g;
  img.spp = 17;
  img.metadata_json = R"({"frame": 42})";
  img.radiance.resize(size_t(w) * h * g.n);
  for (size_t i = 0; i < img.radiance.size(); ++i) img.radiance[i] = float((i * 2654435761u) % 10007) / 1000.f;
  img.depth.resize(size_t(w) * h);
  img.seg_id.resize(size_t(w) * h);
  for (size_t i = 0; i < img.depth.size(); ++i) {
    img.depth[i] = 0.5f * float(i);
    img.seg_id[i] = uint32_t(i * 7 + 1);
  }
  return img;
}
}  // namespace

TEST_CASE("EXR round trip preserves bands, AOVs and attributes") {
  SpectralImage img = random_image(13, 7, BandGrid{380.f, 5.f, 125});
  std::string p = test::tmp_path("roundtrip.exr");
  write_spectral_exr(img, p);
  SpectralImage r = read_spectral_exr(p);
  CHECK(r.width == 13);
  CHECK(r.height == 7);
  CHECK(r.grid.n == 125);
  CHECK(r.grid.lambda_min == 380.f);
  CHECK(r.spp == 17);
  CHECK(r.metadata_json == img.metadata_json);
  CHECK(r.radiance == img.radiance);
  CHECK(r.depth == img.depth);
  CHECK(r.seg_id == img.seg_id);
  // Half precision bands.
  write_spectral_exr(img, p, true);
  SpectralImage h = read_spectral_exr(p);
  for (size_t i = 0; i < img.radiance.size(); ++i)
    CHECK(h.radiance[i] == doctest::Approx(img.radiance[i]).epsilon(1e-3));
  CHECK(band_channel_name(380.f) == "L_0380");
  CHECK(band_channel_name(382.5f) == "L_0382.50");
}

TEST_CASE("NPZ round trip, including forced ZIP64") {
  SpectralImage img = random_image(9, 5, BandGrid{400.f, 10.f, 31});
  for (bool z64 : {false, true}) {
    std::string p = test::tmp_path(z64 ? "rt64.npz" : "rt.npz");
    write_spectral_npz(img, p, z64);
    SpectralImage r = read_spectral_npz(p);
    CHECK(r.radiance == img.radiance);
    CHECK(r.depth == img.depth);
    CHECK(r.seg_id == img.seg_id);
    CHECK(r.grid.n == 31);
    CHECK(r.grid.lambda_min == 400.f);
    CHECK(r.grid.step == 10.f);
    CHECK(r.spp == 17);
    CHECK(r.metadata_json == img.metadata_json);
  }
  const uint8_t abc[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  CHECK(crc32(abc, 9) == 0xCBF43926u);
}
