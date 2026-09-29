// Rendering backend interface (CPU / CUDA-OptiX).
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "spectral/scene/scene.h"

namespace spectral {

struct FilmBuffers {
  int width = 0, height = 0, n_bands = 0;
  std::vector<float> radiance_sum;  // [y][x][band] sum over samples
  std::vector<float> depth;         // [y][x]
  std::vector<uint32_t> seg_id;     // [y][x]
  int spp_done = 0;
  void allocate(int w, int h, int n);
};

using ProgressCallback = std::function<void(int spp_done, int spp_total)>;

class Backend {
 public:
  virtual ~Backend() = default;
  virtual std::string name() const = 0;
  // Build acceleration structures / upload data. The scene must be finalized.
  virtual void prepare(const Scene& scene) = 0;
  // Render samples [first_sample, first_sample + count) for all pixels, adding into film.
  // Writes AOVs when first_sample == 0.
  virtual void render_pass(const Scene& scene, FilmBuffers& film, int first_sample, int count) = 0;
  virtual void set_threads(int /*threads*/) {}
};

bool cuda_backend_available();
// kind: "cpu", "cuda", or "auto" (CUDA if available, else CPU).
std::unique_ptr<Backend> make_backend(const std::string& kind);
std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_cuda_backend();  // throws if not compiled in

}  // namespace spectral
