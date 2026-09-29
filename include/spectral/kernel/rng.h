// PCG32 random number generator with deterministic per-(pixel,sample) seeding.
#pragma once
#include <cstdint>

#include "spectral/kernel/math.h"

namespace spectral {

SPECTRAL_FN uint64_t mix64(uint64_t v) {
  // SplitMix64 finalizer.
  v ^= v >> 30;
  v *= 0xbf58476d1ce4e5b9ULL;
  v ^= v >> 27;
  v *= 0x94d049bb133111ebULL;
  v ^= v >> 31;
  return v;
}

struct Rng {
  uint64_t state = 0x853c49e6748fea9bULL;
  uint64_t inc = 0xda3e39cb94b95bdbULL;

  SPECTRAL_FN void seed(uint64_t init_state, uint64_t seq) {
    state = 0u;
    inc = (seq << 1u) | 1u;
    next_u32();
    state += init_state;
    next_u32();
  }
  SPECTRAL_FN static Rng for_pixel_sample(uint64_t seed, uint32_t px, uint32_t py, uint32_t sample) {
    Rng r;
    uint64_t pixel = (uint64_t(py) << 32) | uint64_t(px);
    r.seed(mix64(seed ^ mix64(pixel + 0x9e3779b97f4a7c15ULL)), mix64(uint64_t(sample) + (seed << 17)));
    return r;
  }
  SPECTRAL_FN uint32_t next_u32() {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + inc;
    uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = uint32_t(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
  }
  // Uniform float in [0, 1).
  SPECTRAL_FN float next1d() { return float(next_u32() >> 8) * (1.f / 16777216.f); }
  SPECTRAL_FN Vec2 next2d() {
    float a = next1d();
    float b = next1d();
    return {a, b};
  }
};

}  // namespace spectral
