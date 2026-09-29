// Jakob-Hanika RGB->spectrum coefficient table management.
#pragma once
#include <memory>
#include <string>

#include "spectral/scene/scene.h"

namespace spectral {

// Returns a cached table. Search order: explicit path, $SPECTRAL_UPLIFT_LUT,
// <data>/uplift/srgb_<res>.coeff, <cache>/srgb_<res>.coeff; otherwise the table is
// generated with the bundled rgb2spec optimizer (sRGB, D65) and written to the cache.
std::shared_ptr<const UpliftLut> obtain_uplift_lut(const std::string& explicit_path = "", int res = 64);
// Runs the optimizer and writes an rgb2spec ".coeff" file.
void generate_uplift_lut(const std::string& output_path, int res);

}  // namespace spectral
