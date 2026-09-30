#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "bvh.h"
#include "spectral/backend/backend.h"
#include "spectral/kernel/film.h"
#include "spectral/kernel/integrator.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace spectral {

void FilmBuffers::allocate(int w, int h, int n) {
  width = w;
  height = h;
  n_bands = n;
  radiance_sum.assign(size_t(w) * h * n, 0.f);
  depth.assign(size_t(w) * h, kInf);
  seg_id.assign(size_t(w) * h, kSkySegId);
  spp_done = 0;
}

namespace {

class CpuBackend final : public Backend {
 public:
  std::string name() const override { return "cpu"; }
  void set_threads(int t) override { threads_ = t; }

  void prepare(const Scene& scene) override {
    view_ = scene.view();
    accel_.build(view_);
  }

  void update(const Scene& scene) override {
    view_ = scene.view();  // arrays may have been reallocated
    accel_.update(view_, uint32_t(scene.meshes().size()));
  }

  void render_pass(const Scene& scene, FilmBuffers& film, int first_sample, int count) override {
    (void)scene;
    const SceneView& s = view_;
    cpu::CpuIntersector isect{&s, &accel_};
    const int W = film.width, H = film.height, tile = 16;
    const int tx = (W + tile - 1) / tile, ty = (H + tile - 1) / tile, ntiles = tx * ty;
    FilmView fv;
    fv.width = W;
    fv.height = H;
    fv.n_bands = film.n_bands;
    fv.radiance = film.radiance_sum.data();
    fv.depth = film.depth.data();
    fv.seg_id = film.seg_id.data();
    int nthreads = threads_ > 0 ? threads_ : int(std::max(1u, std::thread::hardware_concurrency()));
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
#endif
    for (int t = 0; t < ntiles; ++t) {
      int x0 = (t % tx) * tile, y0 = (t / tx) * tile;
      Spectrum L;
      for (int y = y0; y < std::min(y0 + tile, H); ++y) {
        for (int x = x0; x < std::min(x0 + tile, W); ++x) {
          for (int k = 0; k < count; ++k) {
            int sidx = first_sample + k;
            PathAov aov;
            render_sample(s, isect, x, y, uint32_t(sidx), L, sidx == 0 ? &aov : nullptr);
            fv.accumulate(x, y, L);
            if (sidx == 0) {
              fv.depth[size_t(y) * W + x] = aov.depth;
              fv.seg_id[size_t(y) * W + x] = aov.seg_id;
            }
          }
        }
      }
    }
    (void)nthreads;
    film.spp_done += count;
  }

 private:
  SceneView view_;
  cpu::CpuAccel accel_;
  int threads_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

void Backend::preview(const FilmBuffers& film, const PreviewParams& params, std::vector<uint8_t>& rgb) {
  const int W = film.width, H = film.height, n = film.n_bands;
  rgb.resize(size_t(W) * H * 3);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      size_t p = size_t(y) * W + x;
      preview_pixel(film.radiance_sum.data() + p * n, n, film.depth[p], film.seg_id[p], params, rgb.data() + p * 3);
    }
}

void Backend::luminance(const FilmBuffers& film, const float* cmf, float inv_spp, int stride, std::vector<float>& out) {
  const int W = film.width, H = film.height, n = film.n_bands;
  out.clear();
  for (int y = stride / 2; y < H; y += stride)
    for (int x = stride / 2; x < W; x += stride)
      out.push_back(preview_luminance(film.radiance_sum.data() + (size_t(y) * W + x) * n, n, cmf, inv_spp));
}

#if !defined(SPECTRAL_HAS_CUDA)
bool cuda_backend_available() { return false; }
std::unique_ptr<Backend> make_cuda_backend() {
  throw std::runtime_error("CUDA backend not compiled in (configure with -DSPECTRAL_ENABLE_CUDA=ON)");
}
#endif

std::unique_ptr<Backend> make_backend(const std::string& kind) {
  if (kind == "cpu") return make_cpu_backend();
  if (kind == "cuda") return make_cuda_backend();
  if (kind == "auto" || kind.empty()) return cuda_backend_available() ? make_cuda_backend() : make_cpu_backend();
  throw std::runtime_error("Unknown backend '" + kind + "' (cpu | cuda | auto)");
}

}  // namespace spectral
