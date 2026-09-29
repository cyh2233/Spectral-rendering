// Public C++ API of libspectral.
#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "spectral/backend/backend.h"
#include "spectral/scene/scene.h"

namespace spectral {

struct SpectralImage {
  int width = 0, height = 0;
  BandGrid grid;
  int spp = 0;
  std::vector<float> radiance;   // [y][x][band], W / (m^2 sr nm), mean over samples
  std::vector<float> depth;      // [y][x] metres (inf = no hit)
  std::vector<uint32_t> seg_id;  // [y][x] (0xFFFFFFFF = sky / no hit)
  std::string metadata_json = "{}";

  const float* pixel(int x, int y) const { return radiance.data() + (size_t(y) * width + x) * grid.n; }
  std::vector<float> wavelengths() const;
};

struct RenderOptions {
  int spp = -1;             // override scene spp if > 0
  std::string backend;      // override scene backend if non-empty
  int threads = -1;         // override thread count if >= 0
  long long seed = -1;      // override seed if >= 0
  int width = -1, height = -1;
};

class Renderer {
 public:
  Renderer();
  ~Renderer();

  // Load a scene JSON file (see docs/scene_json_schema.md).
  void load_scene_file(const std::string& path);
  // Use a programmatically built scene.
  void set_scene(std::shared_ptr<Scene> scene);
  Scene& scene();

  void set_options(const RenderOptions& o) { options_ = o; }
  void set_progress_callback(ProgressCallback cb) { progress_ = std::move(cb); }
  void cancel() { cancel_ = true; }

  SpectralImage render();

  // Writes EXR / NPZ / preview PNG as configured in `out`.
  static void write_outputs(const SpectralImage& img, const OutputSettings& out);

  static std::string version();

 private:
  std::shared_ptr<Scene> scene_;
  RenderOptions options_;
  ProgressCallback progress_;
  std::atomic<bool> cancel_{false};
};

// Colorimetry helper for the camera pipeline: per-pixel CIE 1931 XYZ ([y][x][3]).
std::vector<float> integrate_to_xyz(const SpectralImage& img);

}  // namespace spectral
