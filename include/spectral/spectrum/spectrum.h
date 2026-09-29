// Dense-band spectral values carried by every path (no hero wavelength).
// A path transports throughput/radiance for all output bands simultaneously.
#pragma once
#include "spectral/kernel/math.h"

namespace spectral {

// Output band grid: band i is centered at lambda_min + i*step and covers
// [center - step/2, center + step/2]. Radiance values are per-nm densities
// (W / (m^2 sr nm)) averaged over the band.
struct BandGrid {
  float lambda_min = 380.f;
  float step = 5.f;
  int n = 125;
  SPECTRAL_FN float center(int i) const { return lambda_min + step * float(i); }
  SPECTRAL_FN float lambda_max() const { return center(n - 1); }
  SPECTRAL_FN int nearest_band(float lambda) const {
    int i = int(floorf((lambda - lambda_min) / step + 0.5f));
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
  }
};

struct Spectrum {
  int n = 0;
  float v[SPECTRAL_MAX_BANDS];

  SPECTRAL_FN Spectrum() {}
  SPECTRAL_FN Spectrum(int bands, float c) : n(bands) {
    for (int i = 0; i < n; ++i) v[i] = c;
  }
  SPECTRAL_FN float operator[](int i) const { return v[i]; }
  SPECTRAL_FN float& operator[](int i) { return v[i]; }

  SPECTRAL_FN Spectrum& operator+=(const Spectrum& o) {
    for (int i = 0; i < n; ++i) v[i] += o.v[i];
    return *this;
  }
  SPECTRAL_FN Spectrum& operator*=(const Spectrum& o) {
    for (int i = 0; i < n; ++i) v[i] *= o.v[i];
    return *this;
  }
  SPECTRAL_FN Spectrum& operator*=(float s) {
    for (int i = 0; i < n; ++i) v[i] *= s;
    return *this;
  }
  // this += a * b * s  (fused accumulate, avoids temporaries on GPU)
  SPECTRAL_FN void add_product(const Spectrum& a, const Spectrum& b, float s) {
    for (int i = 0; i < n; ++i) v[i] += a.v[i] * b.v[i] * s;
  }
  SPECTRAL_FN void add_product3(const Spectrum& a, const Spectrum& b, const Spectrum& c, float s) {
    for (int i = 0; i < n; ++i) v[i] += a.v[i] * b.v[i] * c.v[i] * s;
  }
  SPECTRAL_FN float max_value() const {
    float m = 0.f;
    for (int i = 0; i < n; ++i) m = v[i] > m ? v[i] : m;
    return m;
  }
  SPECTRAL_FN float average() const {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += v[i];
    return n > 0 ? s / float(n) : 0.f;
  }
  SPECTRAL_FN bool is_black() const {
    for (int i = 0; i < n; ++i)
      if (v[i] != 0.f) return false;
    return true;
  }
  SPECTRAL_FN void fill(float c) {
    for (int i = 0; i < n; ++i) v[i] = c;
  }
  SPECTRAL_FN void load(const float* src, int bands) {
    n = bands;
    for (int i = 0; i < n; ++i) v[i] = src[i];
  }
};

SPECTRAL_FN Spectrum operator*(const Spectrum& a, const Spectrum& b) {
  Spectrum r = a;
  r *= b;
  return r;
}
SPECTRAL_FN Spectrum operator*(const Spectrum& a, float s) {
  Spectrum r = a;
  r *= s;
  return r;
}
SPECTRAL_FN Spectrum operator+(const Spectrum& a, const Spectrum& b) {
  Spectrum r = a;
  r += b;
  return r;
}

}  // namespace spectral
