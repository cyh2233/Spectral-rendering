// Host colorimetry helpers: spectral cube / band values -> CIE XYZ, sRGB, CIELAB.
#pragma once
#include <array>
#include <vector>

#include "spectral/spectrum/cie.h"
#include "spectral/spectrum/spectrum.h"

namespace spectral {

using Vec3d = std::array<double, 3>;

// Band-grid colour matching weights: w[c][b] = integral of cmf_c over band b (box).
std::vector<float> band_cmf_weights(const BandGrid& grid, cie::Observer obs = cie::Observer::CIE1931_2deg);

// Integrate band radiances (per-nm densities) to unnormalized XYZ: sum L_b * w_b.
Vec3d bands_to_xyz(const float* bands, const BandGrid& grid, const std::vector<float>& weights);

Vec3d xyz_to_linear_srgb(const Vec3d& xyz);
Vec3d linear_srgb_to_xyz(const Vec3d& rgb);
Vec3d xyz_to_lab(const Vec3d& xyz, const Vec3d& white);
double delta_e76(const Vec3d& lab1, const Vec3d& lab2);
float srgb_encode(float linear);  // linear -> sRGB transfer (gamma)
float srgb_decode(float encoded);

}  // namespace spectral
