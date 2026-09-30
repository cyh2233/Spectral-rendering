// Display preview of the accumulated band cube (host and device). Converts one pixel of the
// radiance-sum film to 8-bit RGB for live viewing: sRGB (CIE 1931), single band, NIR false colour,
// depth or segmentation ids.
#pragma once
#include "spectral/kernel/scene_view.h"

namespace spectral {

enum PreviewMode : int32_t {
  kPreviewSRGB = 0,
  kPreviewBand = 1,      // grayscale of one band
  kPreviewFalseColor = 2,// three bands mapped to R, G, B (e.g. 850 / 650 / 550 nm)
  kPreviewDepth = 3,
  kPreviewSegmentation = 4
};

struct PreviewParams {
  int32_t mode = kPreviewSRGB;
  int32_t band = 0;               // kPreviewBand
  int32_t bands_rgb[3] = {0, 0, 0};  // kPreviewFalseColor
  float scale = 1.f;              // exposure multiplier applied to radiance (already / spp)
  float inv_spp = 1.f;
  float depth_max = 100.f;        // metres mapped to white for kPreviewDepth
  const float* cmf = nullptr;     // 3 * n_bands weights, CIE 1931 (luminance-normalised: Y(1) = 1)
};

SPECTRAL_FN float preview_encode(float x) {
  x = x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
  return x <= 0.0031308f ? 12.92f * x : 1.055f * powf(x, 1.f / 2.4f) - 0.055f;
}

SPECTRAL_FN uint8_t preview_u8(float v) {
  float e = preview_encode(v) * 255.f + 0.5f;
  return uint8_t(e < 0.f ? 0.f : (e > 255.f ? 255.f : e));
}

// Luminance (Y, normalised so a constant spectrum 1 has Y = 1) of one pixel of the sum film.
SPECTRAL_FN float preview_luminance(const float* px, int n, const float* cmf, float inv_spp) {
  float y = 0.f;
  for (int b = 0; b < n; ++b) y += px[b] * cmf[n + b];
  return y * inv_spp;
}

SPECTRAL_FN void preview_pixel(const float* px, int n, float depth, uint32_t seg, const PreviewParams& p,
                               uint8_t* rgb) {
  float r = 0.f, g = 0.f, b = 0.f;
  switch (p.mode) {
    case kPreviewSRGB: {
      float X = 0.f, Y = 0.f, Z = 0.f;
      for (int i = 0; i < n; ++i) {
        X += px[i] * p.cmf[i];
        Y += px[i] * p.cmf[n + i];
        Z += px[i] * p.cmf[2 * n + i];
      }
      float k = p.inv_spp * p.scale;
      X *= k;
      Y *= k;
      Z *= k;
      r = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
      g = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
      b = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;
      break;
    }
    case kPreviewBand:
      r = g = b = px[p.band] * p.inv_spp * p.scale;
      break;
    case kPreviewFalseColor:
      r = px[p.bands_rgb[0]] * p.inv_spp * p.scale;
      g = px[p.bands_rgb[1]] * p.inv_spp * p.scale;
      b = px[p.bands_rgb[2]] * p.inv_spp * p.scale;
      break;
    case kPreviewDepth: {
      float d = depth > 1e30f ? 1.f : depth / p.depth_max;
      d = d > 1.f ? 1.f : d;
      r = g = b = 1.f - d;
      rgb[0] = uint8_t(r * 255.f);
      rgb[1] = uint8_t(g * 255.f);
      rgb[2] = uint8_t(b * 255.f);
      return;
    }
    case kPreviewSegmentation: {
      if (seg == kSkySegId) {
        rgb[0] = rgb[1] = rgb[2] = 0;
        return;
      }
      uint32_t h = seg * 2654435761u + 0x9e3779b9u;
      h ^= h >> 15;
      h *= 2246822519u;
      h ^= h >> 13;
      rgb[0] = uint8_t(64 + (h & 191));
      rgb[1] = uint8_t(64 + ((h >> 8) & 191));
      rgb[2] = uint8_t(64 + ((h >> 16) & 191));
      return;
    }
  }
  rgb[0] = preview_u8(r);
  rgb[1] = preview_u8(g);
  rgb[2] = preview_u8(b);
}

}  // namespace spectral
