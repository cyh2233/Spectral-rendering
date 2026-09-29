// Film buffers (radiance cube + AOVs) written by the integrator drivers.
#pragma once
#include "spectral/kernel/scene_view.h"

namespace spectral {

struct FilmView {
  int32_t width = 0, height = 0, n_bands = 0;
  float* radiance = nullptr;  // [y][x][band], running sum over samples
  float* depth = nullptr;     // [y][x]
  uint32_t* seg_id = nullptr; // [y][x]

  SPECTRAL_FN void accumulate(int x, int y, const Spectrum& L) {
    float* p = radiance + (size_t(y) * width + x) * size_t(n_bands);
    for (int i = 0; i < n_bands; ++i) p[i] += L.v[i];
  }
};

}  // namespace spectral
