#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "spectral/api.h"
#include "spectral/scene/scene_json.h"

namespace spectral::test {

inline std::string data_path(const std::string& rel) { return std::string(SPECTRAL_TEST_DATA_DIR) + "/" + rel; }
inline std::string tmp_path(const std::string& rel) { return std::string(SPECTRAL_TEST_TMP_DIR) + "/" + rel; }

inline SpectralImage render_json(const nlohmann::json& j, int threads = 0) {
  Renderer r;
  r.set_scene(load_scene_json(j, SPECTRAL_TEST_DATA_DIR));
  RenderOptions o;
  o.threads = threads;
  o.backend = "cpu";
  r.set_options(o);
  return r.render();
}

// Mean radiance per band over a pixel rectangle.
inline std::vector<double> mean_bands(const SpectralImage& img, int x0, int y0, int x1, int y1) {
  std::vector<double> m(img.grid.n, 0.0);
  int count = 0;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) {
      const float* p = img.pixel(x, y);
      for (int b = 0; b < img.grid.n; ++b) m[b] += p[b];
      ++count;
    }
  for (double& v : m) v /= count;
  return m;
}

inline double mean_all(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return s / double(v.size());
}

}  // namespace spectral::test
