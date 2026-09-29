#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "spectral/api.h"
#include "spectral/io/image_io.h"
#include "nlohmann/json.hpp"
#include "spectral/scene/scene_json.h"
#include "spectral/spectrum/colorimetry.h"

namespace spectral {

Renderer::Renderer() = default;
Renderer::~Renderer() = default;

std::string Renderer::version() { return "0.1.0"; }

void Renderer::load_scene_file(const std::string& path) { scene_ = load_scene_json_file(path); }
void Renderer::set_scene(std::shared_ptr<Scene> s) { scene_ = std::move(s); }
Scene& Renderer::scene() {
  if (!scene_) throw std::runtime_error("Renderer: no scene loaded");
  return *scene_;
}

SpectralImage Renderer::render() {
  Scene& sc = scene();
  if (options_.width > 0) sc.camera.width = options_.width;
  if (options_.height > 0) sc.camera.height = options_.height;
  if (options_.seed >= 0) sc.params.seed = uint64_t(options_.seed);
  if (!sc.finalized()) sc.finalize();
  int spp = options_.spp > 0 ? options_.spp : sc.render.spp;
  std::string kind = options_.backend.empty() ? sc.render.backend : options_.backend;
  auto backend = make_backend(kind);
  backend->set_threads(options_.threads >= 0 ? options_.threads : sc.render.threads);
  backend->prepare(sc);

  FilmBuffers film;
  film.allocate(sc.camera.width, sc.camera.height, sc.grid.n);
  cancel_ = false;
  int per_pass = std::max(1, sc.render.spp_per_pass);
  for (int s = 0; s < spp && !cancel_; s += per_pass) {
    int n = std::min(per_pass, spp - s);
    backend->render_pass(sc, film, s, n);
    if (progress_) progress_(film.spp_done, spp);
  }
  backend->finish(film);

  SpectralImage img;
  img.width = film.width;
  img.height = film.height;
  img.grid = sc.grid;
  img.spp = film.spp_done;
  img.radiance = std::move(film.radiance_sum);
  float inv = film.spp_done > 0 ? 1.f / float(film.spp_done) : 0.f;
  for (float& v : img.radiance) v *= inv;
  img.depth = std::move(film.depth);
  img.seg_id = std::move(film.seg_id);
  // Metadata: user-provided fields plus what downstream stages (optics, sensor) need.
  nlohmann::json meta = nlohmann::json::parse(sc.output.metadata_json, nullptr, false);
  if (!meta.is_object()) meta = nlohmann::json::object();
  const CameraParams& c = sc.camera;
  nlohmann::json cam_to_world = nlohmann::json::array();
  for (int r = 0; r < 3; ++r) cam_to_world.push_back({c.cam_to_world.m[r][0], c.cam_to_world.m[r][1],
                                                      c.cam_to_world.m[r][2], c.cam_to_world.m[r][3]});
  meta["camera"] = {{"model", "pinhole"}, {"width", c.width}, {"height", c.height},
                    {"tan_half_fov_x", c.tan_half_fov_x}, {"tan_half_fov_y", c.tan_half_fov_y},
                    {"cam_to_world", cam_to_world}, {"convention", "right-handed, +Y up, camera looks down -Z"}};
  meta["spectral"] = {{"lambda_min_nm", sc.grid.lambda_min}, {"step_nm", sc.grid.step}, {"bands", sc.grid.n},
                      {"units", "W/(m^2 sr nm)"}, {"spp", img.spp}, {"renderer", "libspectral " + version()}};
  img.metadata_json = meta.dump();
  return img;
}

void Renderer::write_outputs(const SpectralImage& img, const OutputSettings& out) {
  if (!out.exr.empty()) write_spectral_exr(img, out.exr, out.exr_half);
  if (!out.npz.empty()) write_spectral_npz(img, out.npz);
  if (!out.preview_png.empty()) write_preview_png(img, out.preview_png);
}

std::vector<float> integrate_to_xyz(const SpectralImage& img) {
  std::vector<float> w = band_cmf_weights(img.grid);
  std::vector<float> out(size_t(img.width) * img.height * 3);
  for (size_t p = 0; p < size_t(img.width) * img.height; ++p) {
    Vec3d xyz = bands_to_xyz(img.radiance.data() + p * img.grid.n, img.grid, w);
    for (int c = 0; c < 3; ++c) out[p * 3 + c] = float(xyz[c]);
  }
  return out;
}

}  // namespace spectral
