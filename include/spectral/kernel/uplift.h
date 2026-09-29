// RGB -> spectrum uplifting (Jakob & Hanika 2019, sigmoid-polynomial model).
// Reflectances: r(l) = S(c0 l^2 + c1 l + c2), S(x) = 1/2 + x / (2 sqrt(1 + x^2)).
// NIR policy: beyond `uplift_hold_lambda` the value at the hold wavelength is used.
#pragma once
#include "spectral/kernel/scene_view.h"

namespace spectral {

struct UpliftCoeffs {
  float c0 = 0.f, c1 = 0.f, c2 = 0.f;
};

SPECTRAL_FN float uplift_eval(const UpliftCoeffs& c, float lambda) {
  float x = (c.c0 * lambda + c.c1) * lambda + c.c2;
  if (x > 3.0e38f) return 1.f;
  if (x < -3.0e38f) return 0.f;
  return 0.5f + x / (2.f * sqrtf(1.f + x * x));
}

SPECTRAL_FN int uplift_find_interval(const float* values, int size, float x) {
  int left = 0, last = size - 2, sz = last;
  while (sz > 0) {
    int half = sz >> 1, middle = left + half + 1;
    if (values[middle] <= x) {
      left = middle;
      sz -= half + 1;
    } else {
      sz = half;
    }
  }
  return left < last ? left : last;
}

// rgb in [0,1]^3 (clamped).
SPECTRAL_FN UpliftCoeffs uplift_fetch(const UpliftLutView& lut, float r_, float g_, float b_) {
  float rgb[3] = {clampf(r_, 0.f, 1.f), clampf(g_, 0.f, 1.f), clampf(b_, 0.f, 1.f)};
  UpliftCoeffs out;
  if (rgb[0] == rgb[1] && rgb[1] == rgb[2]) {  // closed form for grays
    float v = rgb[0];
    out.c2 = v <= 0.f ? -8192.f : (v >= 1.f ? 8192.f : (v - 0.5f) / sqrtf(v * (1.f - v)));
    return out;
  }
  if (lut.res <= 1) {  // no table: approximate with luminance gray
    float v = clampf(0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2], 1e-4f, 1.f - 1e-4f);
    out.c2 = (v - 0.5f) / sqrtf(v * (1.f - v));
    return out;
  }
  int res = lut.res;
  int i = 0;
  for (int j = 1; j < 3; ++j)
    if (rgb[j] >= rgb[i]) i = j;
  float z = rgb[i];
  float scale = float(res - 1) / z;
  float x = rgb[(i + 1) % 3] * scale, y = rgb[(i + 2) % 3] * scale;
  uint32_t xi = uint32_t(x) < uint32_t(res - 2) ? uint32_t(x) : uint32_t(res - 2);
  uint32_t yi = uint32_t(y) < uint32_t(res - 2) ? uint32_t(y) : uint32_t(res - 2);
  uint32_t zi = uint32_t(uplift_find_interval(lut.scale, res, z));
  uint32_t offset = (((uint32_t(i) * res + zi) * res + yi) * res + xi) * 3;
  uint32_t dx = 3, dy = 3 * res, dz = 3 * res * res;
  float x1 = x - float(xi), x0 = 1.f - x1, y1 = y - float(yi), y0 = 1.f - y1;
  float z1 = (z - lut.scale[zi]) / (lut.scale[zi + 1] - lut.scale[zi]), z0 = 1.f - z1;
  float c[3];
  for (int j = 0; j < 3; ++j) {
    const float* d = lut.data + offset + j;
    c[j] = ((d[0] * x0 + d[dx] * x1) * y0 + (d[dy] * x0 + d[dy + dx] * x1) * y1) * z0 +
           ((d[dz] * x0 + d[dz + dx] * x1) * y0 + (d[dz + dy] * x0 + d[dz + dy + dx] * x1) * y1) * z1;
  }
  out.c0 = c[0];
  out.c1 = c[1];
  out.c2 = c[2];
  return out;
}

// Linear sRGB reflectance -> band spectrum.
SPECTRAL_FN void uplift_reflectance(const SceneView& s, float r, float g, float b, Spectrum& out) {
  UpliftCoeffs c = uplift_fetch(s.lut, r, g, b);
  out.n = s.grid.n;
  float hold = s.params.uplift_hold_lambda;
  float v_hold = uplift_eval(c, hold);
  for (int i = 0; i < out.n; ++i) {
    float l = s.grid.center(i);
    out.v[i] = l > hold ? v_hold : uplift_eval(c, l);
  }
}

// Linear sRGB radiance (unbounded) -> band spectrum, D65-based illuminant uplifting:
// L(l) = 2m * r_{rgb/2m}(l) * D65n(l), m = max(rgb). RGB (1,1,1) has the luminance of constant 1.
SPECTRAL_FN void uplift_illuminant(const SceneView& s, float r, float g, float b, Spectrum& out) {
  out.n = s.grid.n;
  float m = maxf(r, maxf(g, b));
  if (!(m > 0.f)) {
    out.fill(0.f);
    return;
  }
  float sc = 2.f * m;
  Spectrum refl;
  uplift_reflectance(s, r / sc, g / sc, b / sc, refl);
  for (int i = 0; i < out.n; ++i) out.v[i] = sc * refl.v[i] * s.d65n_bands[i];
}

}  // namespace spectral
