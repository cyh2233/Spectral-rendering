// Device-side preview reduction of the spectral film (regular CUDA compilation, not OptiX).
#include <cuda_runtime.h>

#include "spectral/kernel/preview.h"

using namespace spectral;

namespace {

__global__ void preview_kernel(const float* film, const float* depth, const uint32_t* seg, int w, int h, int n,
                               PreviewParams p, uint8_t* out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  size_t i = size_t(y) * w + x;
  preview_pixel(film + i * n, n, depth[i], seg[i], p, out + i * 3);
}

__global__ void luminance_kernel(const float* film, int w, int h, int n, const float* cmf, float inv_spp, int stride,
                                 int ow, int oh, float* out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= ow || y >= oh) return;
  int sx = stride / 2 + x * stride, sy = stride / 2 + y * stride;
  out[size_t(y) * ow + x] = preview_luminance(film + (size_t(sy) * w + sx) * n, n, cmf, inv_spp);
}

}  // namespace

// Launchers used by cuda_backend.cpp. `p.cmf` must be a device pointer.
cudaError_t spectral_cuda_preview(const float* film, const float* depth, const uint32_t* seg, int w, int h, int n,
                                  const PreviewParams& p, uint8_t* out, cudaStream_t stream) {
  dim3 block(16, 16), grid((w + 15) / 16, (h + 15) / 16);
  preview_kernel<<<grid, block, 0, stream>>>(film, depth, seg, w, h, n, p, out);
  return cudaGetLastError();
}

cudaError_t spectral_cuda_luminance(const float* film, int w, int h, int n, const float* cmf, float inv_spp, int stride,
                                    float* out, int* out_w, int* out_h, cudaStream_t stream) {
  int ow = (w - stride / 2 + stride - 1) / stride, oh = (h - stride / 2 + stride - 1) / stride;
  *out_w = ow;
  *out_h = oh;
  dim3 block(16, 16), grid((ow + 15) / 16, (oh + 15) / 16);
  luminance_kernel<<<grid, block, 0, stream>>>(film, w, h, n, cmf, inv_spp, stride, ow, oh, out);
  return cudaGetLastError();
}
