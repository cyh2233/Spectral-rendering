#include "spectral/session.h"

#include <cmath>
#include <stdexcept>

#include "spectral/spectrum/colorimetry.h"

using nlohmann::json;

namespace spectral {

Session::Session(const std::string& backend, int threads) : backend_kind_(backend), threads_(threads) {}
Session::~Session() = default;

void Session::load_file(const std::string& path) {
  builder_ = make_scene_builder_from_file(path);
  scene_ = builder_->build();
  prepared_ = false;
  dirty_ = true;
}

void Session::load_json(const std::string& text, const std::string& base_dir) {
  builder_ = std::make_unique<SceneBuilder>(json::parse(text, nullptr, true, true), base_dir);
  scene_ = builder_->build();
  prepared_ = false;
  dirty_ = true;
}

Scene& Session::scene() {
  if (!scene_) throw std::runtime_error("Session: no scene loaded");
  return *scene_;
}

std::string Session::backend_name() const { return backend_ ? backend_->name() : backend_kind_; }

void Session::add_asset(const std::string& key, const std::string& path) {
  if (!builder_) throw std::runtime_error("Session: no scene loaded");
  builder_->add_asset(key, path);
}

int Session::spawn(const std::string& instance_json) {
  if (!builder_) throw std::runtime_error("Session: no scene loaded");
  json j = json::parse(instance_json);
  // The pose is not part of the recycling signature.
  json sig_j = j;
  for (const char* k : {"transform", "translate", "rotation", "rotate_deg", "scale"}) sig_j.erase(k);
  std::string sig = sig_j.dump();
  auto it = free_.find(sig);
  if (it != free_.end()) {
    int h = it->second;
    free_.erase(it);
    Object& o = objects_.at(h);
    o.alive = true;
    for (const auto& p : o.parts) scene_->set_instance_hidden(p.instance, false);
    set_transform(h, parse_instance_transform(j));  // pose of the new description
    if (j.contains("seg_id")) set_seg_id(h, j["seg_id"].get<uint32_t>());
    dirty_ = true;
    return h;
  }
  Object o;
  o.signature = sig;
  o.parts = builder_->add_instance(j);
  int h = next_handle_++;
  objects_[h] = std::move(o);
  dirty_ = true;
  return h;
}

void Session::set_transform(int handle, const Affine& to_world) {
  Object& o = objects_.at(handle);
  for (const auto& p : o.parts) scene_->set_instance_transform(p.instance, compose(to_world, p.local));
  dirty_ = true;
}

void Session::set_seg_id(int handle, uint32_t seg) {
  for (const auto& p : objects_.at(handle).parts) scene_->set_instance_seg_id(p.instance, seg);
  dirty_ = true;
}

void Session::remove(int handle) {
  Object& o = objects_.at(handle);
  if (!o.alive) return;
  o.alive = false;
  for (const auto& p : o.parts) scene_->set_instance_hidden(p.instance, true);
  free_.emplace(o.signature, handle);
  dirty_ = true;
}

size_t Session::live_objects() const {
  size_t n = 0;
  for (const auto& [h, o] : objects_) n += o.alive ? 1 : 0;
  return n;
}

void Session::set_camera(const Affine& cam_to_world, float fov_x_deg, int width, int height) {
  CameraParams& c = scene().camera;
  bool resized = c.width != width || c.height != height;
  c.cam_to_world = cam_to_world;
  c.width = width;
  c.height = height;
  c.tan_half_fov_x = std::tan(0.5f * fov_x_deg * kPi / 180.f);
  c.tan_half_fov_y = c.tan_half_fov_x * float(height) / float(width);
  if (resized) film_valid_ = false;
  dirty_ = true;
}

void Session::set_camera_json(const std::string& camera_json) {
  int w = scene().camera.width, h = scene().camera.height;
  builder_->set_camera(json::parse(camera_json));
  if (scene_->camera.width != w || scene_->camera.height != h) film_valid_ = false;
  dirty_ = true;
}

void Session::set_dynamic_lights(const std::string& lights_json) {
  json arr = json::parse(lights_json);
  std::vector<LightRecord> lights;
  for (const auto& l : arr)
    if (l.value("enabled", true)) lights.push_back(builder_->make_light(l));
  scene().set_dynamic_lights(std::move(lights));
  dirty_ = true;
}

void Session::set_sun(Vec3 dir) {
  builder_->set_sun(dir);
  dirty_ = true;
}

void Session::sync() {
  Scene& sc = scene();
  if (!sc.finalized()) sc.finalize();
  if (!backend_) {
    backend_ = make_backend(backend_kind_);
    backend_->set_threads(threads_);
  }
  if (!prepared_) {
    backend_->prepare(sc);
    prepared_ = true;
  } else {
    backend_->update(sc);
  }
  if (cmf_.empty() || int(cmf_.size()) != 3 * sc.grid.n) {
    cmf_ = band_cmf_weights(sc.grid);
    double ysum = 0;
    for (int b = 0; b < sc.grid.n; ++b) ysum += cmf_[sc.grid.n + b];
    for (float& w : cmf_) w = float(w / ysum);  // constant spectrum 1 -> Y = 1
  }
  dirty_ = false;
  film_valid_ = false;  // any scene change restarts accumulation
}

void Session::reset() { film_valid_ = false; }

int Session::render(int spp) {
  Scene& sc = scene();
  if (dirty_ || !sc.finalized()) sync();
  if (!film_valid_ || film_.width != sc.camera.width || film_.height != sc.camera.height ||
      film_.n_bands != sc.grid.n) {
    film_.allocate(sc.camera.width, sc.camera.height, sc.grid.n);
    film_valid_ = true;
  }
  int per_pass = std::max(1, sc.render.spp_per_pass);
  for (int done = 0; done < spp; done += per_pass)
    backend_->render_pass(sc, film_, film_.spp_done, std::min(per_pass, spp - done));
  return film_.spp_done;
}

std::vector<uint8_t> Session::preview(const PreviewOptions& opt) {
  if (!backend_ || film_.spp_done == 0) throw std::runtime_error("Session::preview: nothing rendered yet");
  const BandGrid& g = scene().grid;
  PreviewParams p;
  p.inv_spp = 1.f / float(film_.spp_done);
  p.cmf = cmf_.data();
  p.depth_max = opt.depth_max;
  p.band = g.nearest_band(opt.band_nm);
  for (int k = 0; k < 3; ++k) p.bands_rgb[k] = g.nearest_band(opt.rgb_nm[k]);
  if (opt.mode == "srgb") p.mode = kPreviewSRGB;
  else if (opt.mode == "band") p.mode = kPreviewBand;
  else if (opt.mode == "false_color") p.mode = kPreviewFalseColor;
  else if (opt.mode == "depth") p.mode = kPreviewDepth;
  else if (opt.mode == "seg") p.mode = kPreviewSegmentation;
  else throw std::runtime_error("preview mode must be srgb | band | false_color | depth | seg");
  float scale = 1.f;
  if (opt.auto_exposure && (p.mode == kPreviewSRGB || p.mode == kPreviewBand || p.mode == kPreviewFalseColor)) {
    std::vector<float> lum;
    backend_->luminance(film_, cmf_.data(), p.inv_spp, 4, lum);
    double acc = 0;
    size_t n = 0;
    for (float y : lum)
      if (y > 0.f) {
        acc += std::log(double(y) + 1e-9);
        ++n;
      }
    float key = n ? float(0.18 / std::exp(acc / double(n))) : 1.f;
    // Smooth over frames to avoid flicker in live mode.
    exposure_smoothed_ = exposure_smoothed_ < 0.f ? key : 0.8f * exposure_smoothed_ + 0.2f * key;
    scale = exposure_smoothed_;
  }
  p.scale = scale * std::exp2(opt.exposure_ev);
  std::vector<uint8_t> rgb;
  backend_->preview(film_, p, rgb);
  return rgb;
}

SpectralImage Session::image() {
  if (!backend_) throw std::runtime_error("Session::image: nothing rendered yet");
  backend_->finish(film_);
  SpectralImage img;
  Scene& sc = scene();
  img.width = film_.width;
  img.height = film_.height;
  img.grid = sc.grid;
  img.spp = film_.spp_done;
  img.radiance = film_.radiance_sum;
  float inv = img.spp > 0 ? 1.f / float(img.spp) : 0.f;
  for (float& v : img.radiance) v *= inv;
  img.depth = film_.depth;
  img.seg_id = film_.seg_id;
  json meta = json::parse(sc.output.metadata_json, nullptr, false);
  if (!meta.is_object()) meta = json::object();
  meta["camera"] = {{"model", "pinhole"}, {"width", sc.camera.width}, {"height", sc.camera.height},
                    {"tan_half_fov_x", sc.camera.tan_half_fov_x}, {"tan_half_fov_y", sc.camera.tan_half_fov_y}};
  meta["spectral"] = {{"lambda_min_nm", sc.grid.lambda_min}, {"step_nm", sc.grid.step}, {"bands", sc.grid.n},
                      {"units", "W/(m^2 sr nm)"}, {"spp", img.spp}};
  img.metadata_json = meta.dump();
  return img;
}

}  // namespace spectral
