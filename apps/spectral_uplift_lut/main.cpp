// Generates the Jakob-Hanika sRGB -> spectrum coefficient table.
#include <cstdio>
#include <string>

#include "spectral/spectrum/uplift_lut.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: spectral_uplift_lut <output.coeff> [resolution=64]\n");
    return 1;
  }
  int res = argc > 2 ? std::stoi(argv[2]) : 64;
  spectral::generate_uplift_lut(argv[1], res);
  return 0;
}
