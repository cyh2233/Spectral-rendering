// Persistent rendering session for live / interactive use (e.g. CARLA co-simulation).
//
// The scene is loaded once; afterwards objects are spawned, moved and removed, the camera, the sun
// and dynamic lights change, and each render() call adds samples to a progressive film that is reset
// automatically whenever the scene changed. preview() returns an 8-bit RGB image computed on the
// rendering device; image() downloads the full spectral cube.
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spectral/api.h"
#include "spectral/scene/scene_builder.h"

namespace spectral {

struct PreviewOptions {
  std::string mode = "srgb";        // srgb | band | false_color | depth | seg
  float band_nm = 550.f;            // mode "band"
  float rgb_nm[3] = {850.f, 650.f, 550.f};  // mode "false_color"
  bool auto_exposure = true;        // mean-log-luminance key 0.18, smoothed over frames
  float exposure_ev = 0.f;          // added to the automatic (or fixed, if auto is off) exposure
  float depth_max = 100.f;
};

class Session {
 public:
  // backend: "cpu" | "cuda" | "auto"; threads: CPU threads (0 = all).
  explicit Session(const std::string& backend = "auto", int threads = 0);
  ~Session();

  void load_file(const std::string& scene_json_path);
  void load_json(const std::string& scene_json_text, const std::string& base_dir = ".");
  Scene& scene();
  std::string backend_name() const;

  // --- objects. `instance_json` is an entry of the scene "instances" array; its transform is the
  // initial pose. Returns a handle. Handles of identical descriptions are recycled after remove().
  void add_asset(const std::string& key, const std::string& path);
  int spawn(const std::string& instance_json);
  void set_transform(int handle, const Affine& to_world);
  void set_seg_id(int handle, uint32_t seg_id);
  void remove(int handle);
  size_t live_objects() const;

  // --- camera, lights, sun
  void set_camera(const Affine& cam_to_world, float fov_x_deg, int width, int height);
  void set_camera_json(const std::string& camera_json);
  // JSON array of point / spot / directional lights (scene format); replaces the previous set.
  void set_dynamic_lights(const std::string& lights_json);
  void set_sun(Vec3 direction_to_sun);

  // --- rendering
  void reset();              // restart accumulation
  int render(int spp);       // adds samples; returns the accumulated spp
  int spp() const { return film_.spp_done; }
  std::vector<uint8_t> preview(const PreviewOptions& opt);  // width*height*3, top row first
  SpectralImage image();     // full cube (device -> host copy on GPU)

 private:
  void sync();               // apply pending scene changes to the backend
  std::unique_ptr<SceneBuilder> builder_;
  std::shared_ptr<Scene> scene_;
  std::unique_ptr<Backend> backend_;
  std::string backend_kind_;
  int threads_ = 0;
  FilmBuffers film_;
  bool dirty_ = true, prepared_ = false, film_valid_ = false;
  std::vector<float> cmf_;
  float exposure_smoothed_ = -1.f;

  struct Object {
    std::string signature;
    std::vector<InstancePart> parts;
    bool alive = true;
  };
  std::map<int, Object> objects_;
  std::multimap<std::string, int> free_;  // signature -> removed handle
  int next_handle_ = 1;
};

}  // namespace spectral
