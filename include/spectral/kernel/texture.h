// Texture sampling (bilinear, RGBA8 with optional sRGB decode).
#pragma once
#include "spectral/kernel/scene_view.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace spectral {

SPECTRAL_FN float srgb_to_linear(float c) {
  return c <= 0.04045f ? c * (1.f / 12.92f) : powf((c + 0.055f) * (1.f / 1.055f), 2.4f);
}

SPECTRAL_FN int wrap_coord(int i, int n, int mode) {
  if (mode == kWrapClamp) return i < 0 ? 0 : (i >= n ? n - 1 : i);
  if (mode == kWrapMirror) {
    int period = 2 * n;
    int m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - 1 - m;
  }
  int m = i % n;
  return m < 0 ? m + n : m;
}

SPECTRAL_FN Vec4 texel(const TextureView& t, int x, int y) {
  const uint8_t* p = t.data + (size_t(y) * size_t(t.width) + size_t(x)) * 4;
  Vec4 c(p[0] * (1.f / 255.f), p[1] * (1.f / 255.f), p[2] * (1.f / 255.f), p[3] * (1.f / 255.f));
  if (t.srgb) {
    c.x = srgb_to_linear(c.x);
    c.y = srgb_to_linear(c.y);
    c.z = srgb_to_linear(c.z);
  }
  return c;
}

// uv: glTF convention, (0,0) = top-left of the image.
SPECTRAL_FN Vec4 sample_texture(const SceneView& scene, int32_t tex, Vec2 uv) {
  const TextureView& t = scene.textures[tex];
#if SPECTRAL_DEVICE_CODE
  if (t.gpu_texture) {
    float4 c = tex2D<float4>((cudaTextureObject_t)t.gpu_texture, uv.x, uv.y);
    return Vec4(c.x, c.y, c.z, c.w);
  }
#endif
  float fx = uv.x * float(t.width) - 0.5f, fy = uv.y * float(t.height) - 0.5f;
  float x0f = floorf(fx), y0f = floorf(fy);
  float tx = fx - x0f, ty = fy - y0f;
  int x0 = int(x0f), y0 = int(y0f);
  int xa = wrap_coord(x0, t.width, t.wrap), xb = wrap_coord(x0 + 1, t.width, t.wrap);
  int ya = wrap_coord(y0, t.height, t.wrap), yb = wrap_coord(y0 + 1, t.height, t.wrap);
  Vec4 c00 = texel(t, xa, ya), c10 = texel(t, xb, ya), c01 = texel(t, xa, yb), c11 = texel(t, xb, yb);
  float w00 = (1 - tx) * (1 - ty), w10 = tx * (1 - ty), w01 = (1 - tx) * ty, w11 = tx * ty;
  return Vec4(c00.x * w00 + c10.x * w10 + c01.x * w01 + c11.x * w11,
              c00.y * w00 + c10.y * w10 + c01.y * w01 + c11.y * w11,
              c00.z * w00 + c10.z * w10 + c01.z * w01 + c11.z * w11,
              c00.w * w00 + c10.w * w10 + c01.w * w01 + c11.w * w11);
}

}  // namespace spectral
