// Output/input of spectral images: multichannel OpenEXR, NumPy NPZ, sRGB preview PNG.
#pragma once
#include <map>
#include <string>
#include <vector>

#include "spectral/api.h"

namespace spectral {

// Channel naming: "L_0380" ... for integer band centers, "L_0382.50" otherwise.
std::string band_channel_name(float lambda_nm);

void write_spectral_exr(const SpectralImage& img, const std::string& path, bool half = false);
// Reads an EXR written by write_spectral_exr (band grid from attributes / channel names).
SpectralImage read_spectral_exr(const std::string& path);

struct NpzArray {
  std::string dtype;           // numpy descr, e.g. "<f4", "<u4", "|S12"
  std::vector<size_t> shape;
  std::vector<uint8_t> data;   // raw little-endian bytes
};
void write_npz(const std::string& path, const std::map<std::string, NpzArray>& arrays, bool force_zip64 = false);
std::map<std::string, NpzArray> read_npz(const std::string& path);

void write_spectral_npz(const SpectralImage& img, const std::string& path, bool force_zip64 = false);
SpectralImage read_spectral_npz(const std::string& path);

// Auto-exposed sRGB preview (CIE 1931 integration of the radiance cube).
void write_preview_png(const SpectralImage& img, const std::string& path, float exposure = 0.f);

// Generic RGB(A) float image loading (EXR via tinyexr, HDR/PNG/JPG via stb_image).
struct RgbImage {
  int width = 0, height = 0;
  std::vector<float> rgb;  // linear RGB, [y][x][3]
};
RgbImage load_rgb_image(const std::string& path);

uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc = 0);

}  // namespace spectral
